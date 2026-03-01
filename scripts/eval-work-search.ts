import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { parseArgs } from "node:util";

type EvalFormat = "json" | "jsonl";
type ExpectedMode = "clarify" | "match";

interface WorkSearchRuleSet {
  expected_mode?: ExpectedMode;
  expected_title_contains?: string[];
  expected_writer_contains?: string[];
  expected_winfkey?: string;
  require_prompt_title_echo?: boolean;
  require_prompt_writer_echo?: boolean;
  require_winfkey_for_match?: boolean;
  require_prompt_overlap?: boolean;
  required_substrings?: string[];
  forbidden_substrings?: string[];
}

interface EvalRow {
  id?: string | number;
  question?: string;
  prompt?: string;
  user_input?: string;
  answer?: string;
  response?: string;
  category?: string;
  metadata?: Record<string, unknown>;
}

interface CheckResult {
  name: string;
  passed: boolean;
  detail: string;
}

interface RowResult {
  id: string;
  prompt: string;
  response: string;
  category: string;
  passed: boolean;
  score: number;
  checks: CheckResult[];
}

interface OutputResult {
  inputPath: string;
  rowCount: number;
  evaluatedCount: number;
  passedCount: number;
  averageScore: number;
  rows: RowResult[];
}

const HELP_TEXT = `
Usage:
  pnpm eval:work-search -- --input tmp/evals/supervisor-native.jsonl --output tmp/evals/work-search-rules.json

Input:
  - JSONL or JSON arrays from eval-agent native output
  - RAGAS-shaped output also works if metadata is preserved

Rules source:
  - metadata.work_search_eval
  - metadata.workSearchEval

Supported rule fields:
  - expected_mode: clarify | match
  - expected_title_contains: string[]
  - expected_writer_contains: string[]
  - expected_winfkey: string
  - require_prompt_title_echo: boolean
  - require_prompt_writer_echo: boolean
  - require_winfkey_for_match: boolean
  - require_prompt_overlap: boolean
  - required_substrings: string[]
  - forbidden_substrings: string[]

Options:
  --input <path>             Required. JSON or JSONL evaluator output.
  --output <path>            Required. Where to write the JSON report.
  --format <json|jsonl>      Optional input hint. Auto-detected when omitted.
  --help                     Show this help message.
`;

const parseInput = async (inputPath: string, formatHint?: string): Promise<EvalRow[]> => {
  const raw = await readFile(inputPath, "utf8");
  const trimmed = raw.trim();
  if (!trimmed) {
    return [];
  }

  const inferredFormat: EvalFormat =
    formatHint === "json" || trimmed.startsWith("[") ? "json" : "jsonl";

  if (inferredFormat === "json") {
    const parsed = JSON.parse(trimmed) as unknown;
    if (!Array.isArray(parsed)) {
      throw new Error("JSON input must be an array.");
    }

    return parsed as EvalRow[];
  }

  return trimmed
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as EvalRow);
};

const toRecord = (value: unknown): Record<string, unknown> | undefined =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>) : undefined;

const toStringArray = (value: unknown): string[] | undefined =>
  Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : undefined;

const toNonEmptyString = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim().length > 0 ? value : undefined;

const toBoolean = (value: unknown): boolean | undefined =>
  typeof value === "boolean" ? value : undefined;

const getCategory = (row: EvalRow): string =>
  row.category ??
  (typeof row.metadata?.category === "string" ? row.metadata.category : undefined) ??
  "uncategorized";

const getPrompt = (row: EvalRow): string | undefined =>
  row.question ?? row.prompt ?? row.user_input;

const getResponse = (row: EvalRow): string | undefined => row.answer ?? row.response;

const getRules = (row: EvalRow): WorkSearchRuleSet | undefined => {
  const metadata = toRecord(row.metadata);
  const rawRules = metadata?.work_search_eval ?? metadata?.workSearchEval;
  const rules = toRecord(rawRules);
  if (!rules) {
    return undefined;
  }

  const expectedMode = toNonEmptyString(rules.expected_mode);
  if (expectedMode && !["clarify", "match"].includes(expectedMode)) {
    throw new Error(`Unsupported expected_mode: ${expectedMode}`);
  }

  return {
    expected_mode: expectedMode as ExpectedMode | undefined,
    expected_title_contains: toStringArray(rules.expected_title_contains),
    expected_writer_contains: toStringArray(rules.expected_writer_contains),
    expected_winfkey: toNonEmptyString(rules.expected_winfkey),
    require_prompt_title_echo: toBoolean(rules.require_prompt_title_echo),
    require_prompt_writer_echo: toBoolean(rules.require_prompt_writer_echo),
    require_winfkey_for_match: toBoolean(rules.require_winfkey_for_match),
    require_prompt_overlap: toBoolean(rules.require_prompt_overlap),
    required_substrings: toStringArray(rules.required_substrings),
    forbidden_substrings: toStringArray(rules.forbidden_substrings),
  };
};

const includesAll = (haystack: string, needles: string[]): CheckResult[] =>
  needles.map((needle) => ({
    name: `contains:${needle}`,
    passed: haystack.toLowerCase().includes(needle.toLowerCase()),
    detail: `expected response to contain "${needle}"`,
  }));

const excludesAll = (haystack: string, needles: string[]): CheckResult[] =>
  needles.map((needle) => ({
    name: `forbid:${needle}`,
    passed: !haystack.toLowerCase().includes(needle.toLowerCase()),
    detail: `expected response not to contain "${needle}"`,
  }));

const isClarification = (response: string): boolean =>
  /\?/.test(response) ||
  /(did you mean|please provide|can you tell me more|confirm|which one|if not)/i.test(response);

const extractPromptHints = (prompt: string): { titles: string[]; writers: string[] } => {
  const titles: string[] = [];
  const writers: string[] = [];

  for (const match of prompt.matchAll(/\btitled\s+['"]?([^'",?.]+?)['"]?(?:\s+by\b|[?.!,]|$)/gi)) {
    const title = match[1]?.trim();
    if (title) {
      titles.push(title);
    }
  }

  for (const match of prompt.matchAll(/\bby\s+([a-z0-9 '&.-]+?)(?:[?.!,]|$)/gi)) {
    const writer = match[1]?.trim();
    if (writer) {
      writers.push(writer);
    }
  }

  return {
    titles: [...new Set(titles)],
    writers: [...new Set(writers)],
  };
};

const containsAny = (haystack: string, needles: string[]): boolean =>
  needles.some((needle) => haystack.toLowerCase().includes(needle.toLowerCase()));

const tokenizeOverlapWords = (text: string): string[] =>
  text
    .toLowerCase()
    .replaceAll(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length >= 4);

const evaluateRow = (row: EvalRow): RowResult | undefined => {
  if (getCategory(row) !== "work-search") {
    return undefined;
  }

  const prompt = getPrompt(row);
  const response = getResponse(row);
  const rules = getRules(row);

  if (!prompt || !response || !rules) {
    return undefined;
  }

  const checks: CheckResult[] = [];
  const promptHints = extractPromptHints(prompt);

  if (rules.expected_mode === "clarify") {
    checks.push({
      name: "expected_mode:clarify",
      passed: isClarification(response),
      detail: "expected a clarification-style response",
    });
  }

  if (rules.expected_mode === "match") {
    checks.push({
      name: "expected_mode:match",
      passed: !isClarification(response),
      detail: "expected a non-clarification match response",
    });
  }

  const requirePromptTitleEcho =
    rules.require_prompt_title_echo ??
    (rules.expected_mode === "clarify" && promptHints.titles.length > 0);
  if (requirePromptTitleEcho) {
    checks.push({
      name: "prompt_title_echo",
      passed: containsAny(response, promptHints.titles),
      detail: `expected response to echo one of the requested titles: ${promptHints.titles.join(", ")}`,
    });
  }

  const requirePromptWriterEcho =
    rules.require_prompt_writer_echo ??
    (rules.expected_mode === "clarify" && promptHints.writers.length > 0);
  if (requirePromptWriterEcho) {
    checks.push({
      name: "prompt_writer_echo",
      passed: containsAny(response, promptHints.writers),
      detail: `expected response to echo one of the requested writers: ${promptHints.writers.join(", ")}`,
    });
  }

  if (rules.expected_title_contains) {
    for (const result of includesAll(response, rules.expected_title_contains)) {
      checks.push({ ...result, name: `title:${result.name}` });
    }
  }

  if (rules.expected_writer_contains) {
    for (const result of includesAll(response, rules.expected_writer_contains)) {
      checks.push({ ...result, name: `writer:${result.name}` });
    }
  }

  if (rules.expected_winfkey) {
    checks.push({
      name: "expected_winfkey",
      passed: response.toLowerCase().includes(rules.expected_winfkey.toLowerCase()),
      detail: `expected response to contain WINFKEY "${rules.expected_winfkey}"`,
    });
  }

  const requireWinfkeyForMatch =
    rules.require_winfkey_for_match ?? (rules.expected_mode === "match" && !rules.expected_winfkey);
  if (requireWinfkeyForMatch) {
    checks.push({
      name: "match_requires_winfkey",
      passed: /\bWINF[A-Z0-9-]*\b/i.test(response),
      detail: "expected match response to include a WINFKEY-like identifier",
    });
  }

  if (rules.required_substrings) {
    checks.push(...includesAll(response, rules.required_substrings));
  }

  if (rules.forbidden_substrings) {
    checks.push(...excludesAll(response, rules.forbidden_substrings));
  }

  const requirePromptOverlap = rules.require_prompt_overlap ?? true;
  if (requirePromptOverlap) {
    const promptTokens = tokenizeOverlapWords(prompt);
    checks.push({
      name: "prompt_overlap",
      passed: promptTokens.length === 0 || containsAny(response, promptTokens),
      detail: "expected response to materially overlap with the request wording",
    });
  }

  const passedChecks = checks.filter((check) => check.passed).length;
  const score = checks.length === 0 ? 0 : passedChecks / checks.length;

  return {
    id: String(row.id ?? prompt),
    prompt,
    response,
    category: getCategory(row),
    passed: checks.length > 0 && checks.every((check) => check.passed),
    score,
    checks,
  };
};

const main = async (): Promise<void> => {
  const rawArgs = process.argv.slice(2);
  const args = rawArgs[0] === "--" ? rawArgs.slice(1) : rawArgs;

  const { values } = parseArgs({
    args,
    options: {
      input: { type: "string" },
      output: { type: "string" },
      format: { type: "string" },
      help: { type: "boolean", default: false },
    },
  });

  if (values.help) {
    process.stdout.write(HELP_TEXT);
    return;
  }

  const inputPath = values.input as string | undefined;
  const outputPath = values.output as string | undefined;
  const format = values.format as string | undefined;

  if (!inputPath) {
    throw new Error("Missing required --input path.");
  }
  if (!outputPath) {
    throw new Error("Missing required --output path.");
  }
  if (format && !["json", "jsonl"].includes(format)) {
    throw new Error("--format must be json or jsonl when provided.");
  }

  const rows = await parseInput(inputPath, format);
  const evaluatedRows = rows
    .map((row) => evaluateRow(row))
    .filter((row): row is RowResult => Boolean(row));

  const passedCount = evaluatedRows.filter((row) => row.passed).length;
  const averageScore =
    evaluatedRows.length === 0
      ? 0
      : evaluatedRows.reduce((sum, row) => sum + row.score, 0) / evaluatedRows.length;

  const output: OutputResult = {
    inputPath,
    rowCount: rows.length,
    evaluatedCount: evaluatedRows.length,
    passedCount,
    averageScore,
    rows: evaluatedRows,
  };

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");

  process.stdout.write(`saved: ${outputPath}\n`);
  process.stdout.write(`rows: ${rows.length}\n`);
  process.stdout.write(`evaluated: ${evaluatedRows.length}\n`);
  process.stdout.write(`passed: ${passedCount}\n`);
  process.stdout.write(`average score: ${averageScore.toFixed(3)}\n`);
};

void main().catch((error: unknown) => {
  const message =
    error instanceof Error ? error.message : `Unknown error: ${JSON.stringify(error)}`;
  process.stderr.write(`${message}\n\n${HELP_TEXT}`);
  process.exitCode = 1;
});
