#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";

import { BedrockMultiagentsStack } from "../lib/bedrock-multiagents-stack";

const app = new cdk.App();

new BedrockMultiagentsStack(app, "BedrockMultiagentsStack");
