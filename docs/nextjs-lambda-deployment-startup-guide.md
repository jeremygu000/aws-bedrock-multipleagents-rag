# Next.js Lambda Deployment - Startup Guide

**Date**: 2026-04-13  
**Project Status**: 3 Next.js 15.0.0 apps (chat, web, upload)  
**CDK Status**: Not yet configured (will be added)  
**Decision**: Requires Next.js version upgrade decision

---

## 📊 Current Project Status

### ✅ What We Have

- **3 Independent Next.js 15 Apps**: `apps/chat`, `apps/web`, `apps/upload`
- **pnpm Monorepo**: Workspace configured
- **CDK Infrastructure**: `packages/infra-cdk` exists (Lambda + monitoring)
- **Decision Documentation**: ADR + comparison matrix completed

### ❌ What We Need

- **Next.js Version**: All apps locked to `^15.0.0`
- **Deployment Tool**: Not yet chosen (depends on version upgrade)
- **CloudFront Configuration**: Not yet created
- **Shared Lambda Stack**: Not yet created

---

## 🚀 IMMEDIATE ACTION: Version Upgrade Decision

### Path 1: Upgrade to Next.js 16.2+ (RECOMMENDED)

**Timeline**: 1-2 days | **Implementation**: 8-10 hours

**Step 1: Test Compatibility**

```bash
# Create a test branch
git checkout -b test/nextjs-16-upgrade

# Update packages
pnpm upgrade next@^16.0.0

# Test each app builds
cd apps/chat && pnpm build && cd ../..
cd apps/web && pnpm build && cd ../..
cd apps/upload && pnpm build && cd ../..

# If all builds pass:
# - Run any migration scripts
# - Test local dev: pnpm dev (in each app)
# - Verify no runtime errors

# If issues found:
# - Check Next.js 16 migration guide
# - Fix issues, re-test
```

**Step 2: Commit & Merge**

```bash
# Once verified working
git add package.json pnpm-lock.yaml
git commit -m "upgrade: Next.js 15 → 16.2+ for Lambda deployment"
git push origin test/nextjs-16-upgrade
# → Create PR, get review, merge to main
```

**Result**: ✅ Can use **cdk-nextjs** (AWS Labs, BYOR pattern)

---

### Path 2: Stay on Next.js 15.x (FALLBACK)

**Timeline**: 2-3 weeks | **Implementation**: 15-20 hours

**Steps**:

1. Use **OpenNext** (build tool) to convert Next.js 15 → Lambda zip
2. Manual CDK code to wire 3 apps + CloudFront
3. More complex, but 100% viable

**Result**: ⚠️ Must use **OpenNext + manual CDK** (higher maintenance)

---

## 📋 IMMEDIATE CHECKLIST: This Week

### Day 1-2: Version Decision

- [ ] Run upgrade test (`pnpm upgrade next@^16.0.0`)
- [ ] Test builds for all 3 apps
- [ ] Check for breaking changes
- [ ] Document issues (if any)
- **DECISION**: Commit to upgrade OR stick with 15.x

### Day 3-4: (After Decision)

**IF UPGRADING TO 16.2+**:

- [ ] Merge Next.js 16 to main
- [ ] Install cdk-nextjs package
- [ ] Create SharedNextjsStack skeleton
- [ ] Create example App1Stack

**IF STAYING ON 15.x**:

- [ ] Install @opennextjs/aws package
- [ ] Study OpenNext build process
- [ ] Create CDK Stack for app1 (template)
- [ ] Evaluate additional complexity

---

## 🎯 NEXT PHASE: After Version Decision

### IF Upgrading to Next.js 16.2+ (Recommended Path)

**Phase 2: Architecture Design (1 week)**

```
Week 1:
  Mon-Tue: Design SharedNextjsStack (CloudFront, S3, DynamoDB)
  Wed-Thu: Design App1/2/3 Stacks with BYOR
  Fri: Create CDK code templates
```

**Phase 3: Implementation (1-2 weeks)**

```
Week 2-3:
  Create SharedNextjsStack
  Configure CloudFront (3 cache behaviors)
  Create App1, App2, App3 Stacks
  Deploy to dev environment
  Test routing + caching
```

**Phase 4: Production (1 week)**

```
Week 4:
  Final testing
  Cost validation
  Production deploy
  Monitoring setup
```

**Total Timeline**: ~3-4 weeks to production

---

### IF Staying on Next.js 15.x (Fallback Path)

**Phase 2: OpenNext Configuration (1 week)**

```
Week 1:
  Setup OpenNext build for each app
  Configure CDK for 3 Lambda functions
  Create manual CloudFront config
  Test build/deploy process
```

**Phase 3: Implementation (2-3 weeks)**

```
Week 2-3:
  Build all 3 OpenNext bundles
  Deploy Lambda functions
  Configure API Gateway
  Setup CloudFront routing
```

**Phase 4: Production (1-2 weeks)**

```
Week 4-5:
  Final testing
  Cost validation
  Production deploy
```

**Total Timeline**: ~4-6 weeks to production

---

## 💡 RECOMMENDATION

**Strongly Recommend: Upgrade to Next.js 16.2+**

**Why**:

1. **cdk-nextjs** is purpose-built for this exact scenario
2. Only 1-2 day upgrade vs 2-4 week custom development
3. AWS Labs maintenance = long-term support
4. BYOR pattern exactly matches our multi-app needs
5. Saves ~$17.50/month vs SST (shared CloudFront)

**Risk Level**: 🟢 Low

- Next.js 16 migration is straightforward
- No breaking changes in most apps
- Can test in branch before committing

---

## 📞 Next Steps

### Ready to Decide?

**Option A: Let's test the upgrade**

```bash
# I can run the upgrade test for you now
git checkout -b test/nextjs-16-upgrade
pnpm upgrade next@^16.0.0
pnpm build  # Test all apps
# Report results
```

**Option B: Keep Next.js 15, use OpenNext**

```bash
# I can start OpenNext configuration
# Set up build pipeline for manual CDK deployment
```

**Option C: Need more time to decide**

```bash
# Take the docs and decide later
# We can resume whenever ready
```

---

## 📖 Reference Docs

- `/docs/adr-nextjs-lambda-deployment-2026.md` — Full decision record
- `/docs/nextjs-lambda-deployment-matrix-2026.md` — Tool comparison & checklists
- [Next.js 16 Upgrade Guide](https://nextjs.org/docs/upgrade-guide)
- [cdk-nextjs BYOR Examples](https://github.com/cdklabs/cdk-nextjs/tree/main/examples/bring-your-own)
- [OpenNext AWS Documentation](https://opennext.js.org/aws)
