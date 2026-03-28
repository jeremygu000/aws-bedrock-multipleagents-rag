import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import type { Construct } from "constructs";

export interface MonitoringEc2StackProps extends cdk.StackProps {
  // Free-tier-friendly default stays at t3.micro unless overridden by context.
  instanceType?: string;
  rootVolumeSizeGiB?: number;
  // Restrict this to your public IP (/32) for production-like usage.
  allowedIngressCidr?: string;
  // Keep secret/data resources on stack deletion by default.
  retainDataOnDelete?: boolean;
}

export class MonitoringEc2Stack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: MonitoringEc2StackProps = {}) {
    super(scope, id, props);

    const instanceType = props.instanceType ?? "t3.micro";
    const rootVolumeSizeGiB = props.rootVolumeSizeGiB ?? 8;
    const allowedIngressCidr = props.allowedIngressCidr ?? "0.0.0.0/0";
    const retainDataOnDelete = props.retainDataOnDelete ?? true;
    const dataRemovalPolicy = retainDataOnDelete
      ? cdk.RemovalPolicy.RETAIN
      : cdk.RemovalPolicy.DESTROY;

    // Simple single-AZ public VPC to minimize cost and operational overhead.
    const vpc = new ec2.Vpc(this, "MonitoringVpc", {
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

    const monitoringSg = new ec2.SecurityGroup(this, "MonitoringSecurityGroup", {
      vpc,
      description: "Security group for monitoring EC2 instance",
      allowAllOutbound: true,
    });
    monitoringSg.addIngressRule(ec2.Peer.ipv4(allowedIngressCidr), ec2.Port.tcp(3000), "Grafana");
    monitoringSg.addIngressRule(
      ec2.Peer.ipv4(allowedIngressCidr),
      ec2.Port.tcp(9200),
      "Elasticsearch",
    );

    // Read-only observability permissions let Grafana query AWS metrics/logs/traces.
    const instanceRole = new iam.Role(this, "MonitoringInstanceRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore"),
        iam.ManagedPolicy.fromAwsManagedPolicyName("CloudWatchReadOnlyAccess"),
        iam.ManagedPolicy.fromAwsManagedPolicyName("CloudWatchLogsReadOnlyAccess"),
        iam.ManagedPolicy.fromAwsManagedPolicyName("AWSXrayReadOnlyAccess"),
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonEC2ReadOnlyAccess"),
      ],
    });

    const grafanaAuthSecret = new secretsmanager.Secret(this, "GrafanaAuthSecret", {
      description: "Grafana admin credentials",
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: "admin" }),
        generateStringKey: "password",
        passwordLength: 24,
        excludePunctuation: true,
      },
    });
    grafanaAuthSecret.applyRemovalPolicy(dataRemovalPolicy);
    grafanaAuthSecret.grantRead(instanceRole);

    // Dedicated monitoring instance so Grafana does not compete with Neo4j memory/CPU.
    const instance = new ec2.Instance(this, "MonitoringInstance", {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      role: instanceRole,
      securityGroup: monitoringSg,
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

    // Bootstrap Docker + Grafana and provision a default CloudWatch datasource.
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
      `SECRET_ARN="${grafanaAuthSecret.secretArn}"`,
      `AWS_REGION="${region}"`,
      'SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id "$SECRET_ARN" --region "$AWS_REGION" --query SecretString --output text)',
      'GRAFANA_USERNAME=$(python3 -c \'import json,sys; print(json.loads(sys.argv[1])["username"])\' "$SECRET_JSON")',
      'GRAFANA_PASSWORD=$(python3 -c \'import json,sys; print(json.loads(sys.argv[1])["password"])\' "$SECRET_JSON")',
      "",
      "mkdir -p /opt/grafana/provisioning/datasources",
      "mkdir -p /var/lib/grafana",
      // Grafana container runs as uid/gid 472 by default; enforce writable bind mount permissions.
      "chown -R 472:472 /var/lib/grafana",
      "chmod -R u+rwX,g+rwX /var/lib/grafana",
      "",
      "cat <<'GRAFANA_DATASOURCE' >/opt/grafana/provisioning/datasources/cloudwatch.yml",
      "apiVersion: 1",
      "datasources:",
      "  - name: CloudWatch",
      "    type: cloudwatch",
      "    access: proxy",
      "    isDefault: true",
      "    editable: true",
      "    jsonData:",
      `      defaultRegion: ${region}`,
      "      authType: default",
      "GRAFANA_DATASOURCE",
      "",
      "docker rm -f grafana || true",
      "docker pull grafana/grafana:11.6.0",
      "docker run -d --name grafana --restart unless-stopped \\",
      "  -p 3000:3000 \\",
      '  -e GF_SECURITY_ADMIN_USER="$GRAFANA_USERNAME" \\',
      '  -e GF_SECURITY_ADMIN_PASSWORD="$GRAFANA_PASSWORD" \\',
      "  -e GF_USERS_ALLOW_SIGN_UP=false \\",
      "  -e GF_AUTH_ANONYMOUS_ENABLED=false \\",
      "  -v /var/lib/grafana:/var/lib/grafana \\",
      "  -v /opt/grafana/provisioning:/etc/grafana/provisioning:ro \\",
      "  grafana/grafana:11.6.0",
      "",
      "mkdir -p /var/lib/elasticsearch",
      "chown -R 1000:1000 /var/lib/elasticsearch",
      "docker rm -f elasticsearch || true",
      "docker pull docker.elastic.co/elasticsearch/elasticsearch:8.17.0",
      "docker run -d --name elasticsearch --restart unless-stopped \\",
      "  -p 9200:9200 \\",
      "  -e discovery.type=single-node \\",
      "  -e xpack.security.enabled=false \\",
      "  -e xpack.ml.enabled=false \\",
      "  -e xpack.watcher.enabled=false \\",
      "  -e xpack.graph.enabled=false \\",
      '  -e ES_JAVA_OPTS="-Xms256m -Xmx256m" \\',
      "  -e cluster.routing.allocation.disk.threshold_enabled=false \\",
      "  -v /var/lib/elasticsearch:/usr/share/elasticsearch/data \\",
      "  docker.elastic.co/elasticsearch/elasticsearch:8.17.0",
    );

    new cdk.CfnOutput(this, "MonitoringVpcId", { value: vpc.vpcId });
    new cdk.CfnOutput(this, "MonitoringSecurityGroupId", { value: monitoringSg.securityGroupId });
    new cdk.CfnOutput(this, "MonitoringInstanceId", { value: instance.instanceId });
    new cdk.CfnOutput(this, "MonitoringInstancePublicDns", {
      value: instance.instancePublicDnsName,
    });
    new cdk.CfnOutput(this, "GrafanaHttpUri", {
      value: `http://${instance.instancePublicDnsName}:3000`,
    });
    new cdk.CfnOutput(this, "GrafanaAuthSecretArn", { value: grafanaAuthSecret.secretArn });
    new cdk.CfnOutput(this, "GrafanaDefaultDatasource", { value: "CloudWatch" });
    new cdk.CfnOutput(this, "ElasticsearchHttpUri", {
      value: `http://${instance.instancePublicDnsName}:9200`,
    });
  }
}
