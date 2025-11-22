# Monitoring Alert Rules

## Alert Severity Levels

### HIGH Severity
**Trigger Conditions:**
- PSI > 0.25 (significant distribution shift)
- Entropy change > 30% (major categorical drift)
- Mean shift > 20% (large prediction change)

**Actions:**
1. Send immediate notification to ML team
2. Trigger model retraining pipeline
3. Create detailed drift analysis report
4. Escalate to stakeholders if persistent

**Example:**
```
year drift detected: PSI=2.196, p-value=0.0000
```

### MEDIUM Severity
**Trigger Conditions:**
- 0.1 < PSI < 0.25 (moderate distribution shift)
- 15% < Entropy change < 30%
- 10% < Mean shift < 20%

**Actions:**
1. Add to daily monitoring digest
2. Schedule investigation within 48 hours
3. Monitor for persistence
4. Prepare for potential retraining

**Example:**
```
vehicle_age drift detected: PSI=0.203, p-value=0.0000
```

### LOW Severity
**Trigger Conditions:**
- PSI < 0.1 (minor distribution shift)
- Entropy change < 15%
- Mean shift < 10%

**Actions:**
1. Log to monitoring dashboard
2. Include in weekly summary report
3. No immediate action required

**Example:**
```
trim_grouped drift detected: entropy change=1.2%
```

## Monitoring Schedule

- **Real-time:** Prediction logging
- **Hourly:** Basic metrics aggregation
- **Daily:** Drift detection checks
- **Weekly:** Comprehensive performance report
- **Monthly:** Model performance review

## Retraining Triggers

1. **Automatic Retraining:**
   - 3 consecutive days of HIGH severity alerts
   - Prediction PSI > 0.3
   - Performance degradation > 15%

2. **Scheduled Retraining:**
   - Monthly (regardless of drift)
   - After major data updates
   - Feature schema changes

## Contact Information

- **ML Team:** ml-team@company.com
- **On-call:** #ml-alerts Slack channel
- **Escalation:** VP Engineering
