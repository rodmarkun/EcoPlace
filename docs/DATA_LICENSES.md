# Third-party data

| Source | Pin | License / terms | Use here |
|---|---|---|---|
| Alibaba `cluster-trace-microservices-v2021` and `v2022` ([alibaba/clusterdata](https://github.com/alibaba/clusterdata)) | `0d0f3f1efdbf` | No SPDX license file. The README permits use for study or research purposes and asks that publications be reported to the maintainers. Cite the SoCC '21 (v2021) and v2022 trace papers. | Call-graph structure and call-rate sampling for the workload pipeline. This repository has no trace rows. The Zenodo record has `call_graph_sample.json` (a 16 MB sample of call-graph rows used by the ingestion tests), redistributed for research use under those terms. |
| DeathStarBench ([delimitrou/DeathStarBench](https://github.com/delimitrou/DeathStarBench)) | `6ecb0970` | Apache-2.0 | Call-graph extraction for the application models. |
| TeaStore ([DescartesResearch/TeaStore](https://github.com/DescartesResearch/TeaStore)) | `34b37f7e` | Apache-2.0 | Call-graph extraction for the application models. |
| EdgeSimPy ([EdgeSimPy/EdgeSimPy](https://github.com/EdgeSimPy/EdgeSimPy)) | `76eb5ead` | GPL-3.0 | Historical reference only. The simulator in `sim/` is an independent SimPy implementation and does not include or link EdgeSimPy. |

`scripts/fetch_data.sh` downloads the pinned sources into `data/` (not committed).
