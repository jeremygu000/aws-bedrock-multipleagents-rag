import * as cdk from "aws-cdk-lib";
import * as bedrock from "aws-cdk-lib/aws-bedrock";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as cr from "aws-cdk-lib/custom-resources";
import type { Construct } from "constructs";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const FOUNDATION_MODEL_ID = "amazon.nova-lite-v1:0";

const readText = (filePath: string): string => fs.readFileSync(filePath, "utf8");

const currentFilePath = fileURLToPath(import.meta.url);
const currentDir = path.dirname(currentFilePath);
const infraPackageRoot = path.resolve(currentDir, "../..");
const workspaceRoot = path.resolve(infraPackageRoot, "../..");

const sharedOpenApiRoot = path.join(workspaceRoot, "packages", "shared", "src", "openapi");
const toolWorkDist = path.join(workspaceRoot, "packages", "tool-work-search", "dist");
const toolRagDist = path.join(workspaceRoot, "packages", "tool-rag-search", "dist");
const toolSupervisorDist = path.join(workspaceRoot, "packages", "tool-supervisor", "dist");

export class BedrockAgentsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const workOpenApi = readText(path.join(sharedOpenApiRoot, "work-search.yaml"));
    const ragOpenApi = readText(path.join(sharedOpenApiRoot, "rag-search.yaml"));

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

    const ragSearchFn = new lambda.Function(this, "RagSearchFn", {
      runtime: lambda.Runtime.NODEJS_22_X,
      handler: "handler.handler",
      code: lambda.Code.fromAsset(toolRagDist),
      timeout: cdk.Duration.seconds(30),
      memorySize: 1024,
      tracing: lambda.Tracing.ACTIVE,
      logGroup: ragSearchLogGroup,
    });

    const supervisorToolLogGroup = new logs.LogGroup(this, "SupervisorToolFnLogGroup", {
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

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

    for (const fn of [workSearchFn, ragSearchFn]) {
      fn.addPermission(`${fn.node.id}InvokeByBedrock`, {
        principal: new iam.ServicePrincipal("bedrock.amazonaws.com"),
        action: "lambda:InvokeFunction",
      });
    }

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
    qaCollaborator.node.addDependency(supervisor);
    qaCollaborator.node.addDependency(qaAlias);

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

    new cdk.CfnOutput(this, "SupervisorAliasArn", { value: supervisorAlias.attrAgentAliasArn });
    new cdk.CfnOutput(this, "WorkAliasArn", { value: workAlias.attrAgentAliasArn });
    new cdk.CfnOutput(this, "QaAliasArn", { value: qaAlias.attrAgentAliasArn });
    new cdk.CfnOutput(this, "GatewayFunctionUrl", { value: gatewayUrl.url });
    new cdk.CfnOutput(this, "GatewayFunctionName", { value: gatewayFn.functionName });
  }
}
