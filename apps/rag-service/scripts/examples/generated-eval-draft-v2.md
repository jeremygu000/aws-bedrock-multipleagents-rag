## gen-qa-0001

### Verdict

VALID

### Issues

- Reference should end with punctuation for consistency
- `ground_truth_context` is duplicated at top level and inside `metadata`

### Fix

- Question:
  What is a potential consequence of failure or public criticism for musicians, according to the text?

- Reference:
  Shame and despondency.

### Grounding Truth

Any kind of failure or public criticism can often be something that triggers shame and despondency as well.

### Notes

- Grounding is explicit in the source text. :contentReference[oaicite:0]{index=0}
- Short-answer style is good and much better than the previous version.
- This is effectively a factual/conceptual short-answer item with high evaluation stability.

## gen-qa-0002

### Verdict

VALID (minor refinement)

### Issues

- Question uses "one challenge", which introduces ambiguity
- Reference includes slightly informal phrasing ("whole lot of")

### Fix

- Question:
  What challenge do young artists face in the country music market according to the text?

- Reference:
  They need to create anticipation and make a lot of noise to stand out.

### Grounding Truth

Young artists need to create anticipation and make a whole lot of noise to stand out

### Notes

- Grounding is explicit in the source text :contentReference[oaicite:0]{index=0}
- Good short-answer extraction
- Avoiding "one challenge" improves evaluation stability

## gen-qa-0003

### Verdict

FIXED (previously invalid due to multiple valid answers)

### Issues

- Original version assumed a single correct answer
- Source text contains two valid answers

### Fix

- Question:
  Which albums are nominated for both Album of the Year and Best Pop Vocal Album?

- Reference:
  Midnights and Endless Summer Vacation.

### Grounding Truth

Midnights is also in contention for Best Pop Vocal Album ...
Endless Summer Vacation is also nominated for Album of the Year ... also nominated for Best Pop Vocal Album.

### Notes

- Grounding supported by source text :contentReference[oaicite:0]{index=0}
- Converted to multi-answer format to ensure evaluation stability
- Important fix to avoid false negatives in RAG evaluation

## gen-qa-0004

### Verdict

VALID

### Issues

- Reference should end with punctuation for consistency
- `ground_truth_context` is duplicated in `metadata`

### Fix

- Question:
  What is Zander Hulme's role in the context of the provided text?

- Reference:
  Game composer and sound designer.

### Grounding Truth

Game composer and sound designer Zander Hulme kicks off his 'Creating Ambience' masterclass

### Notes

- Grounding is explicit in the source text. :contentReference[oaicite:0]{index=0}
- This is a clean entity/role extraction item with high evaluation stability.
- Better than the earlier philosophy-based version because it avoids open-ended interpretation.

## gen-qa-0005

### Verdict

VALID

### Issues

- Reference should end with punctuation for consistency
- `ground_truth_context` is duplicated in `metadata`

### Fix

- Question:
  Which film did Amy Bastow co-compose the score for that was directed by James Cameron?

- Reference:
  Deepsea Challenge.

### Grounding Truth

co-composing the score for his 3D feature documentary, Deepsea Challenge

### Notes

- Grounding is explicit in the source text. :contentReference[oaicite:0]{index=0}
- This is a clean single-entity extraction question with a unique answer.
- Strong evaluation stability despite the long surrounding context.

## gen-qa-0006

### Verdict

VALID

### Issues

- Reference should end with punctuation
- `ground_truth_context` duplicated in metadata

### Fix

- Question:
  Which organization has Leah Flanagan been quoted as the National Director of?

- Reference:
  NATSIMO.

### Grounding Truth

NATSIMO National Director, Leah Flanagan

### Notes

- Grounding is explicit in the source text :contentReference[oaicite:0]{index=0}
- Clear entity-role mapping with a single unambiguous answer
- High evaluation stability (no ambiguity, no multiple answers)

## gen-qa-0007

### Verdict

VALID

### Issues

- `ground_truth_context` unnecessarily includes the prefix "Where"
- Reference missing trailing punctuation
- `ground_truth_context` duplicated in metadata

### Fix

- Question:
  Where is the Anatomy of an Album event taking place?

- Reference:
  Spotify HQ - Level 17, 171 Sussex St, Sydney NSW 2000.

### Grounding Truth

Spotify HQ - Level 17, 171 Sussex St, Sydney NSW 2000

### Notes

- Grounding is explicitly stated in the source text
- Clean location extraction with a single unambiguous answer
- Strong evaluation stability (exact match friendly)

## gen-qa-0008

### Verdict

VALID

### Issues

- Reference should end with punctuation
- `ground_truth_context` duplicated in metadata

### Fix

- Question:
  What is the standard notice period required for resigning from APRA membership?

- Reference:
  Six months.

### Grounding Truth

There is a standard six-month notice period for resignation of membership

### Notes

- Grounding is explicitly stated in the source text :contentReference[oaicite:0]{index=0}
- Clean numeric fact extraction with a single unambiguous answer
- High evaluation stability (exact match friendly)

## gen-qa-0009

### Verdict

VALID

### Issues

- Reference should end with punctuation
- `ground_truth_context` is duplicated in `metadata`

### Fix

- Question:
  How many workers from the Philippines, Vietnam and Pakistan provided services to APRA during the Reporting Period?

- Reference: 23.

### Grounding Truth

23 workers located in these countries provided services to APRA.

### Notes

- Grounding is explicit in the source text. :contentReference[oaicite:0]{index=0}
- Clean numeric extraction with a single unambiguous answer.
- High evaluation stability and strong exact-match friendliness.

## gen-qa-0010

### Verdict

VALID (minor refinement needed)

### Issues

- The phrase "the author" is ambiguous because the source contains multiple contributors.
- Reference should end with punctuation.
- `ground_truth_context` is duplicated in metadata.

### Fix

- Question:
  How many times has Vanessa Picken attended SXSW?

- Reference:
  Six.

### Grounding Truth

Number of times at SXSW: Six.

### Notes

- Grounding is explicit in the source text. :contentReference[oaicite:1]{index=1}
- The original version is answerable from the retrieved chunk, but not robust against broader document context.
- Replacing "the author" with the named person improves evaluation stability.

## gen-qa-0011

### Verdict

VALID (minor refinement)

### Issues

- Source text includes "(and more!)", which introduces ambiguity in the answer space
- Reference omits this phrase, which may cause evaluation mismatch

### Fix

- Question:
  Which awards did Julian Wilton and Narayana Johnson win at the Australian Game Developer Awards?

- Reference:
  Best Game and Excellence In Music.

### Grounding Truth

they proceeded to win Best Game and Excellence In Music (and more!) at the Australian Game Developer Awards (AGDA)

### Notes

- Grounding is explicit in the source text :contentReference[oaicite:1]{index=1}
- Reference intentionally excludes "(and more!)" to stabilise evaluation
- Good multi-entity factual extraction with a bounded answer space

## gen-qa-0012

### Verdict

VALID (high quality)

### Issues

- Reference uses capitalised "Collaboration" while source is lowercase
- May cause evaluation mismatch depending on evaluator

### Fix

- Normalize reference to lowercase:
  collaboration, respect, skill, imagination and accountability

### Grounding

Strong exact phrase match in source :contentReference[oaicite:1]{index=1}

### Notes

- Clean fact extraction
- Well-bounded answer space
- Ideal example of "list extraction QA"

## gen-qa-0013

### Verdict

VALID (needs entity normalization)

### Issues

- Reference uses partial entity ("Belinda")
- Full name "Belinda Gehlert" appears earlier in the source

### Risk

- Model may output full name → evaluation mismatch
- Ambiguity risk when dataset scales

### Fix

- Normalize to full entity name:
  belinda gehlert

### Grounding

Exact supporting sentence present :contentReference[oaicite:1]{index=1}

### Notes

- Good entity lookup example
- Important case for enforcing entity normalization rules

## gen-qa-0015

### Verdict

VALID

### Issues

- No P0 issue found
- Optional future improvement: numeric answers may benefit from a normalized numeric field for downstream evaluation

### Risk

- Model may output variant numeric formats such as `8000`, `8,000`, or `$8000`
- Exact-match style evaluators may treat these as mismatches

### Fix

- Keep current reference:
  $8,000

- Optional schema enhancement:
  add `reference_value: 8000`

### Grounding

Exact supporting sentence present :contentReference[oaicite:0]{index=0}

### Notes

- Clean numeric factual QA
- Question and answer scope are fully aligned
- Strong grounding and high evaluation stability

## gen-qa-0016

### Verdict

VALID

### Issues

- No P0 issue found
- Minor: reference is a full sentence while most dataset answers are short phrases (style inconsistency)

### Risk

- Model may output a shorter variant like:
  "went straight to singing"
- Could cause mismatch in strict evaluation (exact match / string similarity)

### Fix

- Preferred normalization (short-answer style):
  gave up four bars of chord introductions and went straight to singing

- Optional (more aggressive normalization):
  went straight to singing

### Grounding

Exact supporting sentence present:
"I gave up four bars of chord introductions and went straight to singing"

### Notes

- Good causal reasoning example (behavior → decision → action)
- Strong grounding and no ambiguity
- Slightly longer answer than dataset norm — consider enforcing short-answer consistency globally

## gen-qa-0017

### Verdict

VALID

### Issues

- No P0 issue found
- Minor: entity normalization should enforce lowercase consistency across dataset

### Risk

- Model may output variants like:
  "Apra Amcos"
  "APRA-AMCOS"
- Could cause mismatch in strict evaluation

### Fix

- Normalize reference to:
  apra amcos

### Grounding

Exact supporting sentence present :contentReference[oaicite:0]{index=0}

### Notes

- Clean entity lookup example
- Strong grounding (explicit "Dean Ormston, APRA AMCOS")
- No ambiguity or multi-entity confusion

## gen-qa-0018

### Verdict

VALID (needs temporal clarity)

### Issues

- Context mixes **multiple years (2026 vs 2027)** with different funding amounts:
  - 2026 chunk: up to $20,000
  - 2027 section: up to $25,000

### Risk

- Model may answer:
  "$25,000" (from later section)
- Leads to false negative in evaluation despite correct reasoning

### Fix

Option A (推荐，改 question)：

- What was the maximum funding amount available for an Annual Arts Grant in 2026?

Option B（保留 question，但更脆弱）：

- Keep reference = $20,000
- But risk remains if retrieval expands

### Grounding

Exact supporting sentence present:
"Grants available up to $20,000..."

### Notes

- Classic **temporal ambiguity bug**
- Important pattern: same entity + different year → different value
- Should be enforced in dataset generation rules

## gen-qa-0019

### Verdict

VALID (needs entity normalization)

### Issues

- `reference` 使用 **非结构化多实体字符串**：
  - `"APRA AMCOS and triple j"`
- 未进行标准化（大小写 / 顺序 / 分隔符）
- 未提供 `reference_list`

### Risk

- 模型输出轻微变化即判错：
  - `"triple j and APRA AMCOS"`（顺序不同）
  - `"APRA AMCOS, triple j"`（分隔符不同）
  - `"APRA AMCOS & triple j"`（符号不同）
- 👉 语义正确但 evaluation fail（典型 false negative）

### Fix

Option A（推荐，结构化）：

````json
"reference_list": [
  "apra amcos",
  "triple j"
]

## gen-qa-0020

### Verdict
VALID (needs answer format normalization)

### Issues
- `reference` 使用 **boolean-style短答案**：
  - `"Yes"`
- ground truth 实际是：
  - `"wheelchair-accessible venue"`（属性型答案）
- 当前是 **Yes/No vs descriptive answer mismatch**

### Risk
- 模型可能输出：
  - `"Yes"` ✅
  - `"Yes, it is wheelchair-accessible"` ❌
  - `"It is a wheelchair-accessible venue"` ❌
- 👉 语义正确但 evaluation fail（高概率）

### Fix
Option A（推荐，改为属性型答案）：
```json
"reference": "wheelchair-accessible"

Option B（保留 Yes/No，但需约束模型）：

强制 short answer:
"Yes" only
evaluator 做：
startswith("yes") 判定
❗但不如 A 稳定


## gen-qa-0021

### Verdict
INVALID (needs source correction and list normalization)

### Issues
- `citation_url` points to a **search page**:
  - `https://www.apraamcos.com.au/search`
- `retrieved_contexts` is a **search snippet / noisy extraction**, not a clean source passage:
  - starts with `work?`
  - contains HTML entity `&amp;`
  - ends with `Load more results`
- `reference` is a **multi-city list stored as a plain string**
- No `reference_list` provided

### Risk
- The search page content is unstable and may change or disappear
- The snippet may be truncated or malformed, leading to false grounding
- Model may output city variants such as:
  - `Sydney, Melbourne, Brisbane, Adelaide and Perth`
  - `Sydney, Melbourne, Brisbane, Adelaide & Perth`
  - different order / separator
- 👉 Semantically correct answer may still fail evaluation

### Fix
Option A（推荐，先修 source）：
- Replace the search-page citation with the canonical APRA AMCOS page that contains the jingle reporting guidance
- Replace `retrieved_contexts` with a clean passage from that page

Option B（同时做 answer normalization）：
```json
"reference_list": [
  "sydney",
  "melbourne",
  "brisbane",
  "adelaide",
  "perth"
]

## gen-qa-0022

### Verdict
VALID (clean factual extraction)

### Issues
- Minor paraphrasing in `reference`:
  - "meeting creative people through events"
  vs
  - "getting to meet so many amazing creative people ... through all the events"
- Slight wording mismatch but meaning is identical

### Risk
- Low risk overall
- Possible model outputs:
  - "meeting creative people through events"
  - "networking with creative people at events"
  - "meeting writers/publishers through events"
- All semantically equivalent → should be accepted

### Fix
Option A（推荐）：
- Keep `reference` as is (concise + correct)

Option B（更严格 eval）：
```json
"reference_list": [
  "royalty collection",
  "meeting creative people through events"
]

## gen-qa-0023

### Verdict
VALID (clean factual constraint)

### Issues
- None significant
- Clear numerical constraint with explicit wording: “10-minute or less”

### Risk
- Very low risk
- Possible model variations:
  - "10 minutes"
  - "10-minute sample"
  - "up to 10 minutes"
- All acceptable if evaluation normalises phrasing

### Fix
Option A（推荐）：
- Keep `reference = "10 minutes"` (clean and precise)

Option B（更严格 eval）：
```json
"reference_list": [
  "10 minutes",
  "10-minute or less"
]
````
