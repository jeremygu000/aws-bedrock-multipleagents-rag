import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import type { Construct } from "constructs";

export interface Neo4jDataStackProps extends cdk.StackProps {
  instanceType?: string;
  rootVolumeSizeGiB?: number;
  volumeSizeGiB?: number;
  allowedIngressCidr?: string;
  retainDataOnDelete?: boolean;
}

export class Neo4jDataStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: Neo4jDataStackProps = {}) {
    super(scope, id, props);

    const instanceType = props.instanceType ?? "t3.small";
    const rootVolumeSizeGiB = props.rootVolumeSizeGiB ?? 8;
    const volumeSizeGiB = props.volumeSizeGiB ?? 20;
    const allowedIngressCidr = props.allowedIngressCidr ?? "0.0.0.0/0";
    const retainDataOnDelete = props.retainDataOnDelete ?? true;
    const dataRemovalPolicy = retainDataOnDelete
      ? cdk.RemovalPolicy.RETAIN
      : cdk.RemovalPolicy.DESTROY;
    const isMicroInstance = instanceType.endsWith(".micro");
    const heapSize = isMicroInstance ? "256M" : "1G";
    const pageCacheSize = isMicroInstance ? "256M" : "1G";

    const vpc = new ec2.Vpc(this, "Neo4jVpc", {
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

    const neo4jSg = new ec2.SecurityGroup(this, "Neo4jSecurityGroup", {
      vpc,
      description: "Security group for Neo4j EC2 instance",
      allowAllOutbound: true,
    });
    neo4jSg.addIngressRule(ec2.Peer.ipv4(allowedIngressCidr), ec2.Port.tcp(7474), "Neo4j HTTP");
    neo4jSg.addIngressRule(ec2.Peer.ipv4(allowedIngressCidr), ec2.Port.tcp(7687), "Neo4j Bolt");

    const instanceRole = new iam.Role(this, "Neo4jInstanceRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore")],
    });

    const neo4jAuthSecret = new secretsmanager.Secret(this, "Neo4jAuthSecret", {
      description: "Neo4j database credentials",
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: "neo4j" }),
        generateStringKey: "password",
        passwordLength: 24,
        excludePunctuation: true,
      },
    });
    neo4jAuthSecret.applyRemovalPolicy(dataRemovalPolicy);
    neo4jAuthSecret.grantRead(instanceRole);

    const instance = new ec2.Instance(this, "Neo4jInstance", {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      role: instanceRole,
      securityGroup: neo4jSg,
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

    const dataVolume = new ec2.CfnVolume(this, "Neo4jDataVolume", {
      availabilityZone: instance.instanceAvailabilityZone,
      size: volumeSizeGiB,
      encrypted: true,
      volumeType: "gp3",
    });
    dataVolume.applyRemovalPolicy(dataRemovalPolicy);

    const dataAttachment = new ec2.CfnVolumeAttachment(this, "Neo4jDataVolumeAttachment", {
      device: "/dev/xvdf",
      instanceId: instance.instanceId,
      volumeId: dataVolume.ref,
    });
    dataAttachment.addDependency(dataVolume);

    const region = cdk.Stack.of(this).region;
    instance.addUserData(
      "set -euxo pipefail",
      "dnf update -y",
      "dnf install -y docker awscli python3",
      "systemctl enable docker",
      "systemctl start docker",
      "",
      "DATA_DEVICE=/dev/xvdf",
      "for i in {1..30}; do",
      '  if [ -b "$DATA_DEVICE" ] || [ -b /dev/nvme1n1 ]; then',
      "    break",
      "  fi",
      "  sleep 2",
      "done",
      'if [ ! -b "$DATA_DEVICE" ] && [ -b /dev/nvme1n1 ]; then DATA_DEVICE=/dev/nvme1n1; fi',
      'if [ ! -b "$DATA_DEVICE" ]; then',
      '  echo "Neo4j data device not found" >&2',
      "  exit 1",
      "fi",
      'if ! blkid "$DATA_DEVICE"; then mkfs -t xfs "$DATA_DEVICE"; fi',
      "mkdir -p /data",
      'if ! grep -q "$DATA_DEVICE /data" /etc/fstab; then',
      '  echo "$DATA_DEVICE /data xfs defaults,nofail 0 2" >> /etc/fstab',
      "fi",
      "mount -a",
      "mkdir -p /data/neo4j/data /data/neo4j/logs /data/neo4j/import /data/neo4j/plugins",
      "",
      `SECRET_ARN="${neo4jAuthSecret.secretArn}"`,
      `AWS_REGION="${region}"`,
      'SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id "$SECRET_ARN" --region "$AWS_REGION" --query SecretString --output text)',
      'NEO4J_USERNAME=$(python3 -c \'import json,sys; print(json.loads(sys.argv[1])["username"])\' "$SECRET_JSON")',
      'NEO4J_PASSWORD=$(python3 -c \'import json,sys; print(json.loads(sys.argv[1])["password"])\' "$SECRET_JSON")',
      "docker rm -f neo4j || true",
      "docker pull neo4j:5-community",
      "docker run -d --name neo4j --restart unless-stopped \\",
      "  -p 7474:7474 -p 7687:7687 \\",
      "  -v /data/neo4j/data:/data \\",
      "  -v /data/neo4j/logs:/logs \\",
      "  -v /data/neo4j/import:/import \\",
      "  -v /data/neo4j/plugins:/plugins \\",
      '  -e NEO4J_AUTH="$NEO4J_USERNAME/$NEO4J_PASSWORD" \\',
      `  -e NEO4J_server_memory_heap_initial__size=${heapSize} \\`,
      `  -e NEO4J_server_memory_heap_max__size=${heapSize} \\`,
      `  -e NEO4J_server_memory_pagecache_size=${pageCacheSize} \\`,
      "  neo4j:5-community",
    );

    new cdk.CfnOutput(this, "Neo4jVpcId", { value: vpc.vpcId });
    new cdk.CfnOutput(this, "Neo4jSecurityGroupId", { value: neo4jSg.securityGroupId });
    new cdk.CfnOutput(this, "Neo4jInstanceId", { value: instance.instanceId });
    new cdk.CfnOutput(this, "Neo4jInstancePublicDns", { value: instance.instancePublicDnsName });
    new cdk.CfnOutput(this, "Neo4jDataVolumeId", { value: dataVolume.ref });
    new cdk.CfnOutput(this, "Neo4jAuthSecretArn", { value: neo4jAuthSecret.secretArn });
    new cdk.CfnOutput(this, "Neo4jBoltUri", {
      value: `bolt://${instance.instancePublicDnsName}:7687`,
    });
    new cdk.CfnOutput(this, "Neo4jHttpUri", {
      value: `http://${instance.instancePublicDnsName}:7474`,
    });
  }
}
