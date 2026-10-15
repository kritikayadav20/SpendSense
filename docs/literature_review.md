# Literature review and verified references

Source pages were checked through web access on the session date, 2026-10-10. The five papers below are from 2022–2024; arXiv entries are labelled as preprints where a proceedings venue was not verified. Abstract-level comparisons identify design lessons, not independent replications. No published score is presented as a SpendSense result.

1. **Liam Toran, Cory Van Der Walt, Alan Sammarone, Alex Keller (2023). “Scalable and Weakly Supervised Bank Transaction Classification.”** arXiv preprint, arXiv:2305.18430. [Verified record](https://arxiv.org/abs/2305.18430), [DOI](https://doi.org/10.48550/arXiv.2305.18430). The work combines weak supervision, transaction representations and neural classification to reduce annotation dependence. SpendSense instead requires documented existing labels; the paper motivates treating label quality and acquisition as central engineering issues.

2. **Antonio J. G. Busson, Rafael Rocha, Rennan Gaio, Rafael Miceli, Ivan Pereira, Daniel de S. Moraes, Sérgio Colcher, Alvaro Veiga, Bruno Rizzi, Francisco Evangelista, Leandro Santos, Fellipe Marques, Marcos Rabaioli, Diego Feldberg, Debora Mattos, João Pasqua, Diogo Dias (2023). “Hierarchical Classification of Financial Transactions Through Context-Fusion of Transformer-based Embeddings and Taxonomy-aware Attention Layer.”** arXiv:2312.07730; record links related proceedings DOI 10.5753/bwaif.2023.229322. [Verified record](https://arxiv.org/abs/2312.07730). The proposed architecture combines merchant/activity descriptors and enforces a hierarchical taxonomy. It motivates preserving source label semantics. SpendSense deliberately implements single-label classical baselines suitable for a small project.

3. **Dimitrios Kotios, Georgios Makridis, Georgios Fatouros, Dimosthenis Kyriazis (2022). “Deep learning enhancing banking services: a hybrid transaction classification and cash flow prediction approach.”** *Journal of Big Data*, 9, article 100. [Publisher and DOI](https://doi.org/10.1186/s40537-022-00651-x). This work connects hybrid transaction categorization with cash-flow modelling for financial management. SpendSense keeps those concerns separate and uses disclosed pace arithmetic instead of claiming a trained forecasting component.

4. **Jaime González-González, Silvia García-Méndez, Francisco de Arriba-Pérez, Francisco J. González-Castaño, Óscar Barba-Seara (2022). “Explainable automatic industrial carbon footprint estimation from bank transaction classification using natural language processing.”** *IEEE Access*, DOI 10.1109/ACCESS.2022.3226324; a later 2024 arXiv deposit is available as arXiv:2405.14505. [Verified author/title record and related DOI](https://arxiv.org/abs/2405.14505). The work studies explanations of transaction classifications for downstream carbon estimates. It supports explaining model evidence and uncertainty; SpendSense's recognized-term display is only feature coverage, not a reproduction of its explanation method.

5. **Pengwei Tang, Huayi Tang, Wenhan Wang, Hanjing Su, Yong Liu (2024). “Decouple then Combine: A Simple and Effective Framework for Fraud Transaction Detection.”** *Proceedings of the 15th Asian Conference on Machine Learning*, PMLR 222:1353–1368. [Publisher record](https://proceedings.mlr.press/v222/tang24a.html). This is adjacent work on fraud detection, not expense categorization. Its attention to class imbalance reinforces the need to inspect minority-class behaviour; SpendSense uses per-class metrics and macro-F1 rather than importing the paper's neural architecture or fraud labels.

## Earlier directly relevant foundation

6. **Silvia García-Méndez, Milagros Fernández-Gavilanes, Jonathan Juncal-Martínez, Francisco J. González-Castaño, Oscar Barba Seara (2020). “Identifying Banking Transaction Descriptions via Support Vector Machine Short-Text Classification Based on a Specialized Labelled Corpus.”** *IEEE Access*, DOI 10.1109/ACCESS.2020.2983584. [Author/title and related DOI record](https://arxiv.org/abs/2404.08664). The 2024 arXiv upload is not a new 2024 journal publication. The work studies SVM classification and similarity-based reduction of short transaction descriptions, directly motivating the linear baseline and duplicate sensitivity analysis.

## Gap addressed by this project

The contribution is an integrated, reproducible student-scale implementation: four controlled TF-IDF baselines, training-only selection, source verification, honest uncertainty and a local budgeting workflow. It is not a claim of a new learning algorithm. Its value for a viva lies in showing how evaluation validity, storage, human correction and accessible presentation fit together.

## Software references

- [scikit-learn text feature extraction](https://scikit-learn.org/1.5/modules/feature_extraction.html#text-feature-extraction)
- [scikit-learn evaluation metrics](https://scikit-learn.org/1.5/modules/model_evaluation.html)
- [scikit-learn pitfalls and leakage](https://scikit-learn.org/1.5/common_pitfalls.html)
- [Streamlit documentation](https://docs.streamlit.io/)
- [SQLite documentation](https://www.sqlite.org/docs.html)
