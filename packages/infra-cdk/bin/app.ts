#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";

// Node ESM requires the emitted .js extension here, or `cdk synth` cannot resolve dist imports.
import { BedrockAgentsStack } from "../lib/bedrock-agents-stack.js";
import { Neo4jDataStack } from "../lib/neo4j-data-stack.js";

const app = new cdk.App();

const neo4jInstanceType = app.node.tryGetContext("neo4jInstanceType") as string | undefined;
const neo4jAllowedIngressCidr = app.node.tryGetContext("neo4jAllowedIngressCidr") as
  | string
  | undefined;
const neo4jRootVolumeSizeContext = app.node.tryGetContext("neo4jRootVolumeSizeGiB") as
  | string
  | number
  | undefined;
const neo4jVolumeSizeContext = app.node.tryGetContext("neo4jVolumeSizeGiB") as
  | string
  | number
  | undefined;
const neo4jRootVolumeSizeGiB =
  neo4jRootVolumeSizeContext === undefined ? undefined : Number(neo4jRootVolumeSizeContext);
const neo4jVolumeSizeGiB =
  neo4jVolumeSizeContext === undefined ? undefined : Number(neo4jVolumeSizeContext);
const neo4jRetainDataContext = app.node.tryGetContext("neo4jRetainDataOnDelete") as
  | string
  | boolean
  | undefined;
const neo4jRetainDataOnDelete =
  neo4jRetainDataContext === undefined
    ? undefined
    : String(neo4jRetainDataContext).toLowerCase() === "true";

new Neo4jDataStack(app, "Neo4jDataStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  instanceType: neo4jInstanceType,
  allowedIngressCidr: neo4jAllowedIngressCidr,
  rootVolumeSizeGiB:
    neo4jRootVolumeSizeGiB !== undefined && Number.isFinite(neo4jRootVolumeSizeGiB)
      ? neo4jRootVolumeSizeGiB
      : undefined,
  volumeSizeGiB:
    neo4jVolumeSizeGiB !== undefined && Number.isFinite(neo4jVolumeSizeGiB)
      ? neo4jVolumeSizeGiB
      : undefined,
  retainDataOnDelete: neo4jRetainDataOnDelete,
});

new BedrockAgentsStack(app, "BedrockAgentsStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});
