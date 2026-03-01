#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";

// Node ESM requires the emitted .js extension here, or `cdk synth` cannot resolve dist imports.
import { BedrockAgentsStack } from "../lib/bedrock-agents-stack.js";

const app = new cdk.App();

new BedrockAgentsStack(app, "BedrockAgentsStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});
