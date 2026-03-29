import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import type { Construct } from "constructs";

export interface PhoenixEc2StackProps extends cdk.StackProps {
  // Free-tier-friendly default stays at t3.micro unless overridden by context.
  instanceType?: string;
  rootVolumeSizeGiB?: number;
  // Restrict this to your public IP (/32) for production-like usage.
  allowedIngressCidr?: string;
  // Keep secret/data resources on stack deletion by default.
  retainDataOnDelete?: boolean;
}

/**
 * Self-hosted Arize Phoenix observability server on a dedicated EC2 instance.
 *
 * Phoenix provides LLM trace collection (OTLP HTTP + gRPC), RAG evaluation
 * metrics, and a web UI for debugging retrieval pipelines.
 *
 * Ports:
 *   6006 — Web UI + OTLP HTTP collector (/v1/traces)
 *   4317 — OTLP gRPC collector
 */
export class PhoenixEc2Stack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: PhoenixEc2StackProps = {}) {
    super(scope, id, props);

    const instanceType = props.instanceType ?? "t3.micro";
    const rootVolumeSizeGiB = props.rootVolumeSizeGiB ?? 8;
    const allowedIngressCidr = props.allowedIngressCidr ?? "0.0.0.0/0";
    const retainDataOnDelete = props.retainDataOnDelete ?? true;
    const dataRemovalPolicy = retainDataOnDelete
      ? cdk.RemovalPolicy.RETAIN
      : cdk.RemovalPolicy.DESTROY;

    // Simple single-AZ public VPC to minimize cost and operational overhead.
    const vpc = new ec2.Vpc(this, "PhoenixVpc", {
      maxAzs: 1,
      natGateways: 0,
      subnetConfiguration: [
        {
          name: "public",
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
      ],
    });

    const phoenixSg = new ec2.SecurityGroup(this, "PhoenixSecurityGroup", {
      vpc,
      description: "Security group for Phoenix observability EC2 instance",
      allowAllOutbound: true,
    });
    phoenixSg.addIngressRule(
      ec2.Peer.ipv4(allowedIngressCidr),
      ec2.Port.tcp(6006),
      "Phoenix UI + OTLP HTTP",
    );
    phoenixSg.addIngressRule(
      ec2.Peer.ipv4(allowedIngressCidr),
      ec2.Port.tcp(4317),
      "Phoenix OTLP gRPC",
    );

    // SSM for remote access; no CloudWatch read policies needed (Phoenix is the observer, not the observed).
    const instanceRole = new iam.Role(this, "PhoenixInstanceRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore")],
    });

    const phoenixAuthSecret = new secretsmanager.Secret(this, "PhoenixAuthSecret", {
      description: "Phoenix admin credentials",
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: "admin@localhost" }),
        generateStringKey: "password",
        passwordLength: 24,
        excludePunctuation: true,
      },
    });
    phoenixAuthSecret.applyRemovalPolicy(dataRemovalPolicy);
    phoenixAuthSecret.grantRead(instanceRole);

    const instance = new ec2.Instance(this, "PhoenixInstance", {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      role: instanceRole,
      securityGroup: phoenixSg,
      requireImdsv2: true,
      blockDevices: [
        {
          deviceName: "/dev/xvda",
          volume: ec2.BlockDeviceVolume.ebs(rootVolumeSizeGiB, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            deleteOnTermination: true,
            encrypted: true,
          }),
        },
      ],
    });

    const region = cdk.Stack.of(this).region;

    // Bootstrap Docker + Arize Phoenix with SQLite persistence and auto-generated admin password.
    instance.addUserData(
      "set -euxo pipefail",
      "dnf update -y",
      "dnf install -y docker awscli python3",
      "systemctl enable docker",
      "systemctl start docker",
      "",
      "if ! swapon --show | grep -q '/swapfile'; then",
      "  if [ ! -f /swapfile ]; then",
      "    fallocate -l 1G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=1024",
      "    chmod 600 /swapfile",
      "    mkswap /swapfile",
      "  fi",
      "  swapon /swapfile || true",
      "  if ! grep -q '^/swapfile ' /etc/fstab; then",
      "    echo '/swapfile none swap sw 0 0' >> /etc/fstab",
      "  fi",
      "fi",
      "",
      `SECRET_ARN="${phoenixAuthSecret.secretArn}"`,
      `AWS_REGION="${region}"`,
      'SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id "$SECRET_ARN" --region "$AWS_REGION" --query SecretString --output text)',
      'PHOENIX_ADMIN_PASSWORD=$(python3 -c \'import json,sys; print(json.loads(sys.argv[1])["password"])\' "$SECRET_JSON")',
      "",
      "mkdir -p /var/lib/phoenix",
      "",
      "docker rm -f phoenix || true",
      "docker pull arizephoenix/phoenix:latest",
      "docker run -d --name phoenix --restart unless-stopped \\",
      "  -p 6006:6006 \\",
      "  -p 4317:4317 \\",
      '  -e PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD="$PHOENIX_ADMIN_PASSWORD" \\',
      "  -e PHOENIX_ENABLE_STRONG_PASSWORD_POLICY=true \\",
      "  -e PHOENIX_DEFAULT_RETENTION_POLICY_DAYS=30 \\",
      "  -e PHOENIX_WORKING_DIR=/mnt/data \\",
      "  -e PHOENIX_TELEMETRY_ENABLED=false \\",
      "  -v /var/lib/phoenix:/mnt/data \\",
      "  arizephoenix/phoenix:latest",
    );

    new cdk.CfnOutput(this, "PhoenixVpcId", { value: vpc.vpcId });
    new cdk.CfnOutput(this, "PhoenixSecurityGroupId", { value: phoenixSg.securityGroupId });
    new cdk.CfnOutput(this, "PhoenixInstanceId", { value: instance.instanceId });
    new cdk.CfnOutput(this, "PhoenixInstancePublicDns", {
      value: instance.instancePublicDnsName,
    });
    new cdk.CfnOutput(this, "PhoenixHttpUri", {
      value: `http://${instance.instancePublicDnsName}:6006`,
    });
    new cdk.CfnOutput(this, "PhoenixOtlpGrpcUri", {
      value: `http://${instance.instancePublicDnsName}:4317`,
    });
    new cdk.CfnOutput(this, "PhoenixAuthSecretArn", { value: phoenixAuthSecret.secretArn });
  }
}
