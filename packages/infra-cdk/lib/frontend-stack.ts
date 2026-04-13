import * as path from "node:path";
import { fileURLToPath } from "node:url";
import * as cdk from "aws-cdk-lib";
import { Duration, Stack, type StackProps } from "aws-cdk-lib";
import {
  Distribution,
  ViewerProtocolPolicy,
  CachePolicy,
  OriginRequestPolicy,
  AllowedMethods,
} from "aws-cdk-lib/aws-cloudfront";
import { HttpOrigin } from "aws-cdk-lib/aws-cloudfront-origins";
import type { Construct } from "constructs";
import { NextjsSite } from "cdk-opennext";

/**
 * Frontend Stack: Multi Zones (3x Next.js apps) behind unified CloudFront
 *
 * Architecture:
 * - CloudFront Distribution (main entry point)
 * - 3x OpenNext Lambda origins (web, chat, upload)
 * - 3x S3 origins (static assets per zone)
 * - CloudFront behaviors route by path:
 *   - / → web zone (default)
 *   - /chat → chat zone
 *   - /upload → upload zone
 *
 * Deployment flow:
 * 1. Build each Next.js app: cd apps/{web,chat,upload} && pnpm build
 * 2. Generate OpenNext output: npx @opennextjs/aws build
 * 3. Deploy CDK: cdk deploy FrontendStack
 *
 * Each zone is deployed as a separate NextjsSite construct with:
 * - Lambda@Edge for server-side rendering (30s timeout)
 * - S3 for static assets (_next/*, public/*)
 * - DynamoDB for ISR revalidation
 * - SQS for revalidation queue
 */
export class FrontendStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const currentDir = path.dirname(fileURLToPath(import.meta.url));
    const infraPackageRoot = path.resolve(currentDir, "../..");
    const workspaceRoot = path.resolve(infraPackageRoot, "../..");

    // =========================================================================
    // Zone 1: Web App
    // =========================================================================
    const webSite = new NextjsSite(this, "WebZone", {
      openNextPath: path.join(workspaceRoot, "apps/web/.open-next"),
      defaultFunctionProps: {
        memorySize: 2048,
        timeout: Duration.seconds(30),
        environment: {
          NODE_ENV: "production",
        },
      },
      // Keep 1 Lambda instance warm to reduce cold starts
      warm: 1,
      warmerInterval: Duration.minutes(5),
      // Disable custom domain for now; we'll use CloudFront alias
      customDomain: undefined,
    });

    // =========================================================================
    // Zone 2: Chat App
    // =========================================================================
    const chatSite = new NextjsSite(this, "ChatZone", {
      openNextPath: path.join(workspaceRoot, "apps/chat/.open-next"),
      defaultFunctionProps: {
        memorySize: 2048,
        timeout: Duration.seconds(30),
        environment: {
          NODE_ENV: "production",
        },
      },
      warm: 1,
      warmerInterval: Duration.minutes(5),
      customDomain: undefined,
    });

    // =========================================================================
    // Zone 3: Upload App
    // =========================================================================
    const uploadSite = new NextjsSite(this, "UploadZone", {
      openNextPath: path.join(workspaceRoot, "apps/upload/.open-next"),
      defaultFunctionProps: {
        memorySize: 2048,
        timeout: Duration.seconds(30),
        environment: {
          NODE_ENV: "production",
        },
      },
      warm: 1,
      warmerInterval: Duration.minutes(5),
      customDomain: undefined,
    });

    // =========================================================================
    // Parent CloudFront Distribution: Route all 3 zones
    // =========================================================================
    // CRITICAL: Each zone must have a unique static asset path to avoid conflicts.
    // OpenNext generates assets to _next/static/ by default, but the construct
    // allows configuring assetPrefix per zone. However, cdk-opennext does not yet
    // support basePath + assetPrefix configuration, so we work around it by
    // creating a parent CloudFront that routes each zone to its own distribution.

    const webDistribution = webSite.distribution;
    const chatDistribution = chatSite.distribution;
    const uploadDistribution = uploadSite.distribution;

    if (!webDistribution || !chatDistribution || !uploadDistribution) {
      throw new Error("OpenNext site distributions not initialized");
    }

    const distribution = new Distribution(this, "FrontendDistribution", {
      comment: "Multi Zones Frontend (Web + Chat + Upload)",
      enableLogging: true,
      logFilePrefix: "cloudfront/",

      // Default behavior: Web Zone (/)
      defaultBehavior: {
        origin: new HttpOrigin(webDistribution.domainName, {
          protocolPolicy: cdk.aws_cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
        }),
        viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: AllowedMethods.ALLOW_ALL,
        compress: true,
        cachePolicy: CachePolicy.CACHING_DISABLED,
        originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
      },

      // Additional behaviors for other zones
      additionalBehaviors: {
        // Chat Zone: /chat*
        "/chat": {
          origin: new HttpOrigin(chatDistribution.domainName, {
            protocolPolicy: cdk.aws_cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
          }),
          viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: AllowedMethods.ALLOW_ALL,
          compress: true,
          cachePolicy: CachePolicy.CACHING_DISABLED,
          originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
        },
        "/chat/*": {
          origin: new HttpOrigin(chatDistribution.domainName, {
            protocolPolicy: cdk.aws_cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
          }),
          viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: AllowedMethods.ALLOW_ALL,
          compress: true,
          cachePolicy: CachePolicy.CACHING_DISABLED,
          originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
        },

        // Upload Zone: /upload*
        "/upload": {
          origin: new HttpOrigin(uploadDistribution.domainName, {
            protocolPolicy: cdk.aws_cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
          }),
          viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: AllowedMethods.ALLOW_ALL,
          compress: true,
          cachePolicy: CachePolicy.CACHING_DISABLED,
          originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
        },
        "/upload/*": {
          origin: new HttpOrigin(uploadDistribution.domainName, {
            protocolPolicy: cdk.aws_cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
          }),
          viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: AllowedMethods.ALLOW_ALL,
          compress: true,
          cachePolicy: CachePolicy.CACHING_DISABLED,
          originRequestPolicy: OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
        },
      },
    });

    // =========================================================================
    // Outputs
    // =========================================================================
    new cdk.CfnOutput(this, "CloudFrontDomainName", {
      value: distribution.domainName,
      description: "CloudFront distribution domain name",
      exportName: "FrontendDistributionDomain",
    });

    new cdk.CfnOutput(this, "WebZoneDomain", {
      value: `https://${distribution.domainName}/`,
      description: "Web zone home",
    });

    new cdk.CfnOutput(this, "ChatZoneDomain", {
      value: `https://${distribution.domainName}/chat`,
      description: "Chat zone root",
    });

    new cdk.CfnOutput(this, "UploadZoneDomain", {
      value: `https://${distribution.domainName}/upload`,
      description: "Upload zone root",
    });

    new cdk.CfnOutput(this, "WebZoneDistribution", {
      value: webDistribution.domainName,
      description: "Web zone OpenNext distribution (for debugging)",
    });

    new cdk.CfnOutput(this, "ChatZoneDistribution", {
      value: chatDistribution.domainName,
      description: "Chat zone OpenNext distribution (for debugging)",
    });

    new cdk.CfnOutput(this, "UploadZoneDistribution", {
      value: uploadDistribution.domainName,
      description: "Upload zone OpenNext distribution (for debugging)",
    });
  }
}
