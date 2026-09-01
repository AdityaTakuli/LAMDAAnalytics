# Model training results

| Folder | Supervision | Description |
| --- | --- | --- |
| `country_month/` | Country-month aggregate inbound | Archived regression + classification runs |
| `pair_pooled/` | Directed pair-month (340 links) | Pooled bilateral India↔VN, US↔CN, etc. |

Run pair training:

```bash
python train_pair_models.py --config config.yaml --task both --overwrite
```
