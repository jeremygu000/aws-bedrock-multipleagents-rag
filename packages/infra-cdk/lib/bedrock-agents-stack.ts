import * as cdk from "aws-cdk-lib";
import * as bedrock from "aws-cdk-lib/aws-bedrock";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as lambdaEventSources from "aws-cdk-lib/aws-lambda-event-sources";
import { PythonFunction, PythonLayerVersion } from "@aws-cdk/aws-lambda-python-alpha";
import * as logs from "aws-cdk-lib/aws-logs";
import * as opensearch from "aws-cdk-lib/aws-opensearchservice";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as s3n from "aws-cdk-lib/aws-s3-notifications";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as sqs from "aws-cdk-lib/aws-sqs";
import * as cr from "aws-cdk-lib/custom-resources";
import type { Construct } from "constructs";
import * as fs from "node:fs";
import * as path from "node:path";
import { env as processEnv } from "node:process";
import { fileURLToPath } from "node:url";

const FOUNDATION_MODEL_ID = "amazon.nova-lite-v1:0";

/**
 * Read a UTF-8 text file from disk.
 *
 * This helper is used for loading OpenAPI schema payloads
 * that are embedded in Bedrock action group definitions.
 */
const readText = (filePath: string): string => fs.readFileSync(filePath, "utf8");

const currentFilePath = fileURLToPath(import.meta.url);
const currentDir = path.dirname(currentFilePath);
const infraPackageRoot = path.resolve(currentDir, "../..");
const workspaceRoot = path.resolve(infraPackageRoot, "../..");

const sharedOpenApiRoot = path.join(workspaceRoot, "packages", "shared", "src", "openapi");
const toolWorkDist = path.join(workspaceRoot, "packages", "tool-work-search", "dist");
const toolSupervisorDist = path.join(workspaceRoot, "packages", "tool-supervisor", "dist");
const ragPythonEntry = path.join(workspaceRoot, "apps", "rag-service");

/**
 * Main application stack for Bedrock agents and action-group Lambda executors.
 *
 * Responsibilities:
 * - Provision guardrails and versions.
 * - Provision action-group Lambdas (work search + Python RAG).
 * - Provision supervisor gateway and memory table.
 * - Wire collaborator aliases for Bedrock supervisor routing.
 */
export class BedrockAgentsStack extends cdk.Stack {
  /**
   * Create all Bedrock agents, Lambda tools, and cross-resource permissions.
   *
   * @param scope CDK construct scope.
   * @param id Logical stack identifier.
   * @param props Optional stack properties such as env/account/region.
   */
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Load OpenAPI contracts that define Bedrock action-group request/response shapes.
    const workOpenApi = readText(path.join(sharedOpenApiRoot, "work-search.yaml"));
    const ragOpenApi = readText(path.join(sharedOpenApiRoot, "rag-search.yaml"));

    // Guardrail for lower-risk work search interactions.
    const workGuardrail = new bedrock.CfnGuardrail(this, "WorkGuardrailLow", {
      name: "work-guardrail-low",
      blockedInputMessaging:
        "[work-guardrail-low][input-block] Sorry, I can't help with that request.",
      blockedOutputsMessaging:
        "[work-guardrail-low][output-block] Sorry, I can't provide that response.",
      contentPolicyConfig: {
        filtersConfig: [
          { type: "HATE", inputStrength: "MEDIUM", outputStrength: "MEDIUM" },
          { type: "INSULTS", inputStrength: "MEDIUM", outputStrength: "MEDIUM" },
          { type: "SEXUAL", inputStrength: "MEDIUM", outputStrength: "MEDIUM" },
          { type: "VIOLENCE", inputStrength: "MEDIUM", outputStrength: "MEDIUM" },
          { type: "PROMPT_ATTACK", inputStrength: "HIGH", outputStrength: "NONE" },
        ],
      },
    });

    const workGuardrailV1 = new bedrock.CfnGuardrailVersion(this, "WorkGuardrailLowV1", {
      guardrailIdentifier: workGuardrail.attrGuardrailId,
      description: "v1 low restriction for work search",
    });

    // Stricter guardrail for APRA-domain Q&A responses.
    const qaGuardrail = new bedrock.CfnGuardrail(this, "QaGuardrailStrict", {
      name: "qa-guardrail-strict",
      blockedInputMessaging:
        "[qa-guardrail-strict][input-block] Guardrail triggered. I can help with APRA AMCOS licensing, membership, or work registration questions.",
      blockedOutputsMessaging:
        "[qa-guardrail-strict][output-block] Guardrail triggered. I can only provide APRA AMCOS licensing, membership, or work registration guidance.",
      contentPolicyConfig: {
        filtersConfig: [
          { type: "HATE", inputStrength: "LOW", outputStrength: "LOW" },
          { type: "INSULTS", inputStrength: "LOW", outputStrength: "LOW" },
          { type: "SEXUAL", inputStrength: "LOW", outputStrength: "LOW" },
          { type: "VIOLENCE", inputStrength: "LOW", outputStrength: "LOW" },
          { type: "MISCONDUCT", inputStrength: "LOW", outputStrength: "LOW" },
          { type: "PROMPT_ATTACK", inputStrength: "LOW", outputStrength: "NONE" },
        ],
      },
      topicPolicyConfig: {
        topicsConfig: [
          {
            name: "Copyright evasion",
            type: "DENY",
            definition: "Requests about bypassing licensing or avoiding royalty payments.",
            examples: [
              "How do I avoid music licence fees?",
              "How can I play music without paying APRA?",
            ],
          },
        ],
      },
      sensitiveInformationPolicyConfig: {
        piiEntitiesConfig: [
          { type: "EMAIL", action: "BLOCK" },
          { type: "PHONE", action: "BLOCK" },
          { type: "CREDIT_DEBIT_CARD_NUMBER", action: "BLOCK" },
        ],
      },
    });

    const qaGuardrailV1 = new bedrock.CfnGuardrailVersion(this, "QaGuardrailStrictV1", {
      guardrailIdentifier: qaGuardrail.attrGuardrailId,
      description: "v1 strict for APRA QA",
    });

    // Work search action Lambda (Node.js implementation).
    const workSearchLogGroup = new logs.LogGroup(this, "WorkSearchFnLogGroup", {
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const workSearchFn = new lambda.Function(this, "WorkSearchFn", {
      runtime: lambda.Runtime.NODEJS_22_X,
      handler: "handler.handler",
      code: lambda.Code.fromAsset(toolWorkDist),
      timeout: cdk.Duration.seconds(30),
      memorySize: 512,
      tracing: lambda.Tracing.ACTIVE,
      logGroup: workSearchLogGroup,
    });

    const ragSearchLogGroup = new logs.LogGroup(this, "RagSearchFnLogGroup", {
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const openSearchVolumeSizeGiB = Number(processEnv.RAG_OPENSEARCH_VOLUME_GIB ?? "10");
    const ragSearchDomain = new opensearch.Domain(this, "RagSearchDomain", {
      version: opensearch.EngineVersion.OPENSEARCH_2_19,
      capacity: {
        dataNodes: 1,
        dataNodeInstanceType: processEnv.RAG_OPENSEARCH_INSTANCE_TYPE ?? "t3.small.search",
      },
      ebs: {
        enabled: true,
        volumeSize:
          Number.isFinite(openSearchVolumeSizeGiB) && openSearchVolumeSizeGiB > 0
            ? openSearchVolumeSizeGiB
            : 10,
        volumeType: ec2.EbsDeviceVolumeType.GP3,
      },
      zoneAwareness: {
        enabled: false,
      },
      enforceHttps: true,
      nodeToNodeEncryption: true,
      encryptionAtRest: {
        enabled: true,
      },
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Environment map for Python RAG Lambda. Secret ARN is preferred over plaintext password.
    const ragSearchFnEnv: Record<string, string> = {
      RAG_DB_HOST: processEnv.RAG_DB_HOST ?? processEnv.RDS_HOST ?? "",
      RAG_DB_PORT: processEnv.RAG_DB_PORT ?? processEnv.RDS_PORT ?? "5432",
      RAG_DB_NAME: processEnv.RAG_DB_NAME ?? processEnv.RDS_DB_NAME ?? "postgres",
      RAG_DB_USER: processEnv.RAG_DB_USER ?? processEnv.RDS_MASTER_USERNAME ?? "postgres",
      RAG_DB_PASSWORD_SECRET_ARN:
        processEnv.RAG_DB_PASSWORD_SECRET_ARN ?? processEnv.RDS_MASTER_SECRET_ARN ?? "",
      RAG_DB_PASSWORD_SECRET_JSON_KEY: processEnv.RAG_DB_PASSWORD_SECRET_JSON_KEY ?? "password",
      RAG_DB_SSLMODE: processEnv.RAG_DB_SSLMODE ?? "require",
      RAG_EMBED_DIM: processEnv.RAG_EMBED_DIM ?? "1024",
      RAG_SPARSE_BACKEND: processEnv.RAG_SPARSE_BACKEND ?? "opensearch",
      RAG_OPENSEARCH_ENDPOINT:
        processEnv.RAG_OPENSEARCH_ENDPOINT ?? `https://${ragSearchDomain.domainEndpoint}`,
      RAG_OPENSEARCH_INDEX: processEnv.RAG_OPENSEARCH_INDEX ?? "kb_chunks",
      RAG_OPENSEARCH_TIMEOUT_S: processEnv.RAG_OPENSEARCH_TIMEOUT_S ?? "10",
      RAG_ANSWER_MODEL_ID: processEnv.RAG_ANSWER_MODEL_ID ?? FOUNDATION_MODEL_ID,
      RAG_ANSWER_MAX_TOKENS: processEnv.RAG_ANSWER_MAX_TOKENS ?? "500",
      RAG_ANSWER_TEMPERATURE: processEnv.RAG_ANSWER_TEMPERATURE ?? "0.05",
      QWEN_API_KEY: processEnv.QWEN_API_KEY ?? processEnv.DASHSCOPE_API_KEY ?? "",
      QWEN_API_KEY_SECRET_ARN: processEnv.QWEN_API_KEY_SECRET_ARN ?? "",
      QWEN_API_KEY_SECRET_KEY: processEnv.QWEN_API_KEY_SECRET_KEY ?? "DASHSCOPE_API_KEY",
      QWEN_MODEL_ID: processEnv.QWEN_MODEL_ID ?? processEnv.LLM_MODEL ?? "qwen-plus",
      QWEN_EMBEDDING_MODEL_ID: processEnv.QWEN_EMBEDDING_MODEL_ID ?? "text-embedding-v3",
      QWEN_BASE_URL:
        processEnv.QWEN_BASE_URL ?? "https://dashscope.aliyuncs.com/compatible-mode/v1",
      QWEN_MAX_TOKENS: processEnv.QWEN_MAX_TOKENS ?? "500",
      QWEN_TEMPERATURE: processEnv.QWEN_TEMPERATURE ?? "0.0",
      RAG_ROUTE_MIN_HITS: processEnv.RAG_ROUTE_MIN_HITS ?? "3",
      RAG_ROUTE_TOP_SCORE_THRESHOLD: processEnv.RAG_ROUTE_TOP_SCORE_THRESHOLD ?? "0.015",
      RAG_ROUTE_COMPLEX_QUERY_TOKEN_THRESHOLD:
        processEnv.RAG_ROUTE_COMPLEX_QUERY_TOKEN_THRESHOLD ?? "18",
      RAG_ENABLE_QUERY_REWRITE: processEnv.RAG_ENABLE_QUERY_REWRITE ?? "true",
      RAG_ENABLE_HYBRID_RETRIEVAL: processEnv.RAG_ENABLE_HYBRID_RETRIEVAL ?? "true",
      RAG_S3_BUCKET: processEnv.RAG_S3_BUCKET ?? "",
      RAG_INGESTION_QUEUE_URL: processEnv.RAG_INGESTION_QUEUE_URL ?? "",
      RAG_CHUNK_SIZE: processEnv.RAG_CHUNK_SIZE ?? "512",
      RAG_CHUNK_OVERLAP: processEnv.RAG_CHUNK_OVERLAP ?? "64",
      RAG_CHUNK_MIN_SIZE: processEnv.RAG_CHUNK_MIN_SIZE ?? "50",
      RAG_EMBED_BATCH_SIZE: processEnv.RAG_EMBED_BATCH_SIZE ?? "20",
      RAG_MAX_UPLOAD_SIZE_MB: processEnv.RAG_MAX_UPLOAD_SIZE_MB ?? "50",
    };

    // Explicit local override only. Avoid injecting plaintext password by default.
    if (processEnv.RAG_DB_PASSWORD) {
      ragSearchFnEnv.RAG_DB_PASSWORD = processEnv.RAG_DB_PASSWORD;
    }

    // Directories and caches excluded from PythonFunction Docker bundling.
    const pythonAssetExcludes = [
      ".venv",
      ".pytest_cache",
      ".ruff_cache",
      "__pycache__",
      "tests",
      "layers",
      "hybrid_rag_service.egg-info",
      ".mypy_cache",
    ];

    // RAG action Lambda now points to Python service entrypoint.
    const ragSearchFn = new PythonFunction(this, "RagSearchFn", {
      runtime: lambda.Runtime.PYTHON_3_12,
      entry: ragPythonEntry,
      index: "lambda_tool.py",
      handler: "handler",
      timeout: cdk.Duration.seconds(30),
      memorySize: 1024,
      tracing: lambda.Tracing.ACTIVE,
      logGroup: ragSearchLogGroup,
      environment: ragSearchFnEnv,
      bundling: { assetExcludes: pythonAssetExcludes },
    });
    ragSearchDomain.grantReadWrite(ragSearchFn);

    const ragDbPasswordSecretArn = ragSearchFnEnv.RAG_DB_PASSWORD_SECRET_ARN;
    if (ragDbPasswordSecretArn) {
      // Grant runtime secret-read permission only when a secret ARN is provided.
      const ragDbPasswordSecret = secretsmanager.Secret.fromSecretCompleteArn(
        this,
        "RagDbPasswordSecret",
        ragDbPasswordSecretArn,
      );
      ragDbPasswordSecret.grantRead(ragSearchFn);
    }

    const qwenApiKeySecretArn = ragSearchFnEnv.QWEN_API_KEY_SECRET_ARN;
    if (qwenApiKeySecretArn) {
      // Grant read access for optional Qwen API key secret.
      const qwenApiKeySecret = secretsmanager.Secret.fromSecretCompleteArn(
        this,
        "QwenApiKeySecret",
        qwenApiKeySecretArn,
      );
      qwenApiKeySecret.grantRead(ragSearchFn);
    }

    new cdk.CfnOutput(this, "RagOpenSearchEndpoint", {
      value: `https://${ragSearchDomain.domainEndpoint}`,
      description: "OpenSearch domain endpoint for BM25 sparse retrieval.",
    });

    // ── Document Ingestion Pipeline ──────────────────────────────────────────

    // S3 bucket for document uploads.
    const ingestionBucket = new s3.Bucket(this, "IngestionBucket", {
      bucketName: processEnv.RAG_S3_BUCKET || undefined,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: false,
      lifecycleRules: [
        {
          prefix: "uploads/",
          expiration: cdk.Duration.days(90),
        },
      ],
    });

    // Dead-letter queue for failed ingestion messages.
    const ingestionDlq = new sqs.Queue(this, "IngestionDLQ", {
      retentionPeriod: cdk.Duration.days(14),
      visibilityTimeout: cdk.Duration.seconds(300),
    });

    // Main ingestion queue with DLQ after 3 receive attempts.
    const ingestionQueue = new sqs.Queue(this, "IngestionQueue", {
      visibilityTimeout: cdk.Duration.seconds(300),
      retentionPeriod: cdk.Duration.days(4),
      deadLetterQueue: {
        queue: ingestionDlq,
        maxReceiveCount: 3,
      },
    });

    // Trigger ingestion queue for every object created under uploads/.
    ingestionBucket.addEventNotification(
      s3.EventType.OBJECT_CREATED,
      new s3n.SqsDestination(ingestionQueue),
      { prefix: "uploads/" },
    );

    const ingestionLogGroup = new logs.LogGroup(this, "IngestionFnLogGroup", {
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Lambda Layer: heavy document-parsing deps (pymupdf, python-docx, bs4, lxml).
    const docParsingLayer = new PythonLayerVersion(this, "DocParsingLayer", {
      entry: path.join(ragPythonEntry, "layers", "doc-parsing"),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_12],
      description: "Document parsing dependencies (pymupdf, python-docx, bs4, lxml)",
    });

    // Ingestion Lambda: processes documents from SQS, writes embeddings to OpenSearch.
    const ingestionFn = new PythonFunction(this, "IngestionFn", {
      runtime: lambda.Runtime.PYTHON_3_12,
      entry: ragPythonEntry,
      index: "ingestion_handler.py",
      handler: "handler",
      timeout: cdk.Duration.minutes(5),
      memorySize: 1024,
      tracing: lambda.Tracing.ACTIVE,
      logGroup: ingestionLogGroup,
      layers: [docParsingLayer],
      environment: {
        ...ragSearchFnEnv,
        RAG_S3_BUCKET: ingestionBucket.bucketName,
        RAG_INGESTION_QUEUE_URL: ingestionQueue.queueUrl,
        RAG_CHUNK_SIZE: processEnv.RAG_CHUNK_SIZE ?? "512",
        RAG_CHUNK_OVERLAP: processEnv.RAG_CHUNK_OVERLAP ?? "64",
        RAG_CHUNK_MIN_SIZE: processEnv.RAG_CHUNK_MIN_SIZE ?? "50",
        RAG_EMBED_BATCH_SIZE: processEnv.RAG_EMBED_BATCH_SIZE ?? "20",
        RAG_MAX_UPLOAD_SIZE_MB: processEnv.RAG_MAX_UPLOAD_SIZE_MB ?? "50",
      },
      bundling: { assetExcludes: pythonAssetExcludes },
    });

    // Wire SQS as the Lambda event source (one document per invocation).
    ingestionFn.addEventSource(
      new lambdaEventSources.SqsEventSource(ingestionQueue, {
        batchSize: 1,
        reportBatchItemFailures: true,
      }),
    );

    // IAM: ingestion Lambda reads uploaded docs and writes embeddings to OpenSearch.
    ingestionBucket.grantRead(ingestionFn);
    ragSearchDomain.grantReadWrite(ingestionFn);

    // IAM: ragSearchFn needs S3 read/write for sync-mode upload endpoint.
    ingestionBucket.grantReadWrite(ragSearchFn);

    // Secrets access for ingestion Lambda (mirrors ragSearchFn secret grants).
    if (ragDbPasswordSecretArn) {
      const ragDbPasswordSecret2 = secretsmanager.Secret.fromSecretCompleteArn(
        this,
        "IngestionDbPasswordSecret",
        ragDbPasswordSecretArn,
      );
      ragDbPasswordSecret2.grantRead(ingestionFn);
    }
    if (qwenApiKeySecretArn) {
      const qwenApiKeySecret2 = secretsmanager.Secret.fromSecretCompleteArn(
        this,
        "IngestionQwenApiKeySecret",
        qwenApiKeySecretArn,
      );
      qwenApiKeySecret2.grantRead(ingestionFn);
    }

    const supervisorToolLogGroup = new logs.LogGroup(this, "SupervisorToolFnLogGroup", {
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Session memory table used by the gateway Lambda for conversation continuity.
    const memoryTable = new dynamodb.Table(this, "SupervisorMemoryTable", {
      partitionKey: { name: "sessionId", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      timeToLiveAttribute: "ttl",
    });

    const gatewayFn = new lambda.Function(this, "SupervisorGatewayFn", {
      runtime: lambda.Runtime.NODEJS_22_X,
      handler: "handler.handler",
      code: lambda.Code.fromAsset(toolSupervisorDist),
      timeout: cdk.Duration.seconds(120),
      memorySize: 512,
      tracing: lambda.Tracing.ACTIVE,
      logGroup: supervisorToolLogGroup,
      environment: {
        RERANK_MODEL_ARN: "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        RERANK_REGION: "us-west-2",
        MEMORY_TABLE_NAME: memoryTable.tableName,
        DEPLOY_TIMESTAMP: Date.now().toString(),
      },
    });

    memoryTable.grantReadWriteData(gatewayFn);

    // Allow gateway to call Bedrock reranking and runtime APIs.
    gatewayFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["bedrock:Rerank", "bedrock-agent-runtime:Rerank", "bedrock-agent-runtime:*"],
        resources: ["*"],
      }),
    );

    gatewayFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["bedrock:InvokeModel"],
        resources: ["*"],
      }),
    );

    // Let Bedrock invoke both action-group Lambdas.
    for (const fn of [workSearchFn, ragSearchFn]) {
      fn.addPermission(`${fn.node.id}InvokeByBedrock`, {
        principal: new iam.ServicePrincipal("bedrock.amazonaws.com"),
        action: "lambda:InvokeFunction",
      });
    }

    /**
     * Create a Bedrock service role that can invoke models and action-group Lambdas.
     *
     * @param name Logical role id.
     * @returns IAM role for a Bedrock agent.
     */
    const createAgentRole = (name: string) =>
      new iam.Role(this, name, {
        assumedBy: new iam.ServicePrincipal("bedrock.amazonaws.com"),
        inlinePolicies: {
          AgentPolicy: new iam.PolicyDocument({
            statements: [
              new iam.PolicyStatement({
                actions: [
                  "bedrock:InvokeModel",
                  "bedrock:InvokeModelWithResponseStream",
                  "bedrock:ApplyGuardrail",
                ],
                resources: ["*"],
              }),
              new iam.PolicyStatement({
                actions: ["lambda:InvokeFunction"],
                resources: [workSearchFn.functionArn, ragSearchFn.functionArn],
              }),
            ],
          }),
        },
      });

    const workAgentRole = createAgentRole("WorkAgentRole");
    const qaAgentRole = createAgentRole("QaAgentRole");
    const supervisorRole = createAgentRole("SupervisorRole");

    // Specialist agent for work search use cases.
    const workAgent = new bedrock.CfnAgent(this, "WorkAgent", {
      agentName: "work-search-agent",
      foundationModel: FOUNDATION_MODEL_ID,
      agentResourceRoleArn: workAgentRole.roleArn,
      autoPrepare: true,
      guardrailConfiguration: {
        guardrailIdentifier: workGuardrail.attrGuardrailId,
        guardrailVersion: workGuardrailV1.attrVersion,
      },
      instruction: [
        "You are a Work Search agent.",
        "Goal: find the correct work using title, writer, ISWC, ISRC, or publishers.",
        "Ask at most two short clarification questions if needed.",
        "When ready, call the work_search action.",
        "Return concise structured results.",
      ].join("\n"),
      actionGroups: [
        {
          actionGroupName: "work_search",
          actionGroupState: "ENABLED",
          actionGroupExecutor: { lambda: workSearchFn.functionArn },
          apiSchema: { payload: workOpenApi },
        },
        {
          actionGroupName: "UserInput",
          parentActionGroupSignature: "AMAZON.UserInput",
          actionGroupState: "ENABLED",
        },
      ],
    });

    const workAlias = new bedrock.CfnAgentAlias(this, "WorkAlias", {
      agentAliasName: "live",
      agentId: workAgent.attrAgentId,
    });

    // Specialist agent for APRA grounded Q&A via rag_search.
    const qaAgent = new bedrock.CfnAgent(this, "QaAgent", {
      agentName: "apra-qa-agent",
      foundationModel: FOUNDATION_MODEL_ID,
      agentResourceRoleArn: qaAgentRole.roleArn,
      autoPrepare: true,
      guardrailConfiguration: {
        guardrailIdentifier: qaGuardrail.attrGuardrailId,
        guardrailVersion: qaGuardrailV1.attrVersion,
      },
      instruction: [
        "You are an APRA AMCOS domain Q&A agent.",
        "Only answer APRA AMCOS related questions.",
        "Always call rag_search to retrieve grounded information.",
        "If evidence is insufficient, ask for clarification or refuse.",
        "Always include citations returned from rag_search.",
      ].join("\n"),
      actionGroups: [
        {
          actionGroupName: "rag_search",
          actionGroupState: "ENABLED",
          actionGroupExecutor: { lambda: ragSearchFn.functionArn },
          apiSchema: { payload: ragOpenApi },
        },
        {
          actionGroupName: "UserInput",
          parentActionGroupSignature: "AMAZON.UserInput",
          actionGroupState: "ENABLED",
        },
      ],
    });

    const qaAlias = new bedrock.CfnAgentAlias(this, "QaAlias", {
      agentAliasName: "live",
      agentId: qaAgent.attrAgentId,
    });

    // Multi-agent collaboration requires the supervisor service role to read and invoke collaborator aliases.
    supervisorRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["bedrock:GetAgentAlias", "bedrock:InvokeAgent"],
        resources: [workAlias.attrAgentAliasArn, qaAlias.attrAgentAliasArn],
      }),
    );

    // Supervisor router orchestrates collaborator agents.
    const supervisor = new bedrock.CfnAgent(this, "SupervisorAgent", {
      agentName: "supervisor-agent",
      foundationModel: FOUNDATION_MODEL_ID,
      agentResourceRoleArn: supervisorRole.roleArn,
      agentCollaboration: "SUPERVISOR_ROUTER",
      skipResourceInUseCheckOnDelete: true,
      // Multi-agent setup must add collaborators to the DRAFT first, then prepare or deploy it.
      autoPrepare: false,
      instruction: [
        "You are a routing supervisor.",
        "Route work search requests to WorkSearchAgent.",
        "Route APRA AMCOS domain questions to ApraQaAgent.",
        "If the request is ambiguous, ask one short clarification question.",
        "Otherwise refuse politely.",
      ].join("\n"),
    });

    const workCollaborator = new cr.AwsCustomResource(this, "AssociateWorkCollaborator", {
      installLatestAwsSdk: true,
      onCreate: {
        service: "BedrockAgent",
        action: "associateAgentCollaborator",
        parameters: {
          agentId: supervisor.attrAgentId,
          agentVersion: "DRAFT",
          collaboratorName: "WorkSearchAgent",
          collaborationInstruction:
            "Handle work search requests and return concise structured matches.",
          relayConversationHistory: "TO_COLLABORATOR",
          agentDescriptor: {
            aliasArn: workAlias.attrAgentAliasArn,
          },
        },
        physicalResourceId: cr.PhysicalResourceId.of("WorkSearchAgentCollaborator"),
      },
      onDelete: {
        service: "BedrockAgent",
        action: "disassociateAgentCollaborator",
        parameters: {
          agentId: supervisor.attrAgentId,
          agentVersion: "DRAFT",
          collaboratorName: "WorkSearchAgent",
        },
      },
      policy: cr.AwsCustomResourcePolicy.fromSdkCalls({
        resources: cr.AwsCustomResourcePolicy.ANY_RESOURCE,
      }),
    });
    // Ensure collaborator association happens after supervisor/alias creation.
    workCollaborator.node.addDependency(supervisor);
    workCollaborator.node.addDependency(workAlias);

    const qaCollaborator = new cr.AwsCustomResource(this, "AssociateQaCollaborator", {
      installLatestAwsSdk: true,
      onCreate: {
        service: "BedrockAgent",
        action: "associateAgentCollaborator",
        parameters: {
          agentId: supervisor.attrAgentId,
          agentVersion: "DRAFT",
          collaboratorName: "ApraQaAgent",
          collaborationInstruction:
            "Handle APRA AMCOS questions with grounded retrieval and returned citations.",
          relayConversationHistory: "TO_COLLABORATOR",
          agentDescriptor: {
            aliasArn: qaAlias.attrAgentAliasArn,
          },
        },
        physicalResourceId: cr.PhysicalResourceId.of("ApraQaAgentCollaborator"),
      },
      onDelete: {
        service: "BedrockAgent",
        action: "disassociateAgentCollaborator",
        parameters: {
          agentId: supervisor.attrAgentId,
          agentVersion: "DRAFT",
          collaboratorName: "ApraQaAgent",
        },
      },
      policy: cr.AwsCustomResourcePolicy.fromSdkCalls({
        resources: cr.AwsCustomResourcePolicy.ANY_RESOURCE,
      }),
    });
    // Ensure collaborator association happens after supervisor/alias creation.
    qaCollaborator.node.addDependency(supervisor);
    qaCollaborator.node.addDependency(qaAlias);

    // Live alias for supervisor is created after collaborators are associated.
    const supervisorAlias = new bedrock.CfnAgentAlias(this, "SupervisorAlias", {
      agentAliasName: "live",
      agentId: supervisor.attrAgentId,
    });
    supervisorAlias.node.addDependency(workCollaborator);
    supervisorAlias.node.addDependency(qaCollaborator);

    // Gateway Lambda: grant InvokeAgent permission for supervisor and inject agent IDs.
    gatewayFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["bedrock:InvokeAgent"],
        resources: [supervisorAlias.attrAgentAliasArn],
      }),
    );
    gatewayFn.addEnvironment("SUPERVISOR_AGENT_ID", supervisor.attrAgentId);
    gatewayFn.addEnvironment("SUPERVISOR_ALIAS_ID", supervisorAlias.attrAgentAliasId);

    const gatewayUrl = gatewayFn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    // Stack outputs are used by local scripts and test clients.
    new cdk.CfnOutput(this, "SupervisorAliasArn", { value: supervisorAlias.attrAgentAliasArn });
    new cdk.CfnOutput(this, "WorkAliasArn", { value: workAlias.attrAgentAliasArn });
    new cdk.CfnOutput(this, "QaAliasArn", { value: qaAlias.attrAgentAliasArn });
    new cdk.CfnOutput(this, "GatewayFunctionUrl", { value: gatewayUrl.url });
    new cdk.CfnOutput(this, "GatewayFunctionName", { value: gatewayFn.functionName });
    new cdk.CfnOutput(this, "IngestionBucketName", {
      value: ingestionBucket.bucketName,
      description: "S3 bucket for document uploads.",
    });
    new cdk.CfnOutput(this, "IngestionQueueUrl", {
      value: ingestionQueue.queueUrl,
      description: "SQS queue URL for async ingestion.",
    });
    new cdk.CfnOutput(this, "IngestionDLQUrl", {
      value: ingestionDlq.queueUrl,
      description: "Dead letter queue for failed ingestion messages.",
    });
  }
}
