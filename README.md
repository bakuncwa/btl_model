# Low Performing Product in Sephora Dataset Product Recommendation Model

**Dataset:** [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

**Event:** Between the Lines: Machine Learning Study Jam – April 4, 2025

**Developed by:** Gabrielle Ysabel Almirol, Chief Technology Officer, Association of Information Management Benilde

---

## Project Description

Low Performing Product in Sephora Dataset Product Recommendation Model is a machine learning solution developed for the **Between the Lines: Machine Learning Study Jam**. The model predicts customer purchasing behavior based on **ingredient concentration matching** with key chemical ingredients in **low-performing sales revenue product P442990** (REN Clean Skincare's *Clean Screen Mineral SPF 30 Mattifying Face Sunscreen*) and **seasonal skincare needs**.

---

## Tech Stack

| Category                     | Tools / Libraries                                      |
|-------------------------------|-------------------------------------------------------|
| **Programming Language**      | Python 3.10                                          |
| **ML Framework**              | CatBoostClassifier                                   |
| **Data Processing**           | pandas, numpy, Apache Spark                           |
| **Visualization**             | matplotlib, seaborn                                  |
| **Modeling**                  | catboost, scikit-learn                                |
| **Development Tools**         | Jupyter Notebook, Python, Flask (for API)           |

---

## Ingredient Analysis & Benefits

### 1. **Humectants** (e.g., Glycerin, Hyaluronic Acid, Panthenol)

* Attract moisture into the skin
* Maintain hydration and reduce flakiness
* Reinforce natural moisture barrier

### 2. **Emollients** (e.g., Squalane, Ceramides, Shea Butter)

* Soften and smooth skin texture
* Restore lipid barrier
* Prevent cold-induced irritation

### 3. **Occlusive Agents** (e.g., Dimethicone, Petrolatum, Beeswax)

* Form a protective seal to lock in moisture
* Shield skin from wind and low humidity
* Extend humectant/emollient effects

### 4. **Antioxidants** (e.g., Vitamin E, Green Tea Extract, Niacinamide)

* Reduce oxidative stress and inflammation
* Support skin repair
* Boost moisturizing effectiveness

### 5. **UV & Pollution Protectants** (e.g., Zinc Oxide, Titanium Dioxide, Licorice Root Extract)

* Guard against UV even in winter
* Minimize irritation and pigmentation
* Aid in seasonal skin recovery

---

## Model-Based Ingredient Optimization

Utilizing `CatBoostClassifier`, the model predicts customer preference likelihood using ingredient concentrations in **P442990** and similar products during **cold months (Jan–Apr)**.

| Label | Definition                                       |
| ----- | ------------------------------------------------ |
| `1`   | Higher-than-median concentration match (> 0.075) |
| `0`   | Lower-than-median concentration match (≤ 0.075)  |

### Classification Report

| Category       | Precision | Recall | F1-Score | Support |
| -------------- | --------- | ------ | -------- | ------- |
| 0 (Low Match)  | 1.00      | 1.00   | 1.00     | 61,020  |
| 1 (High Match) | 1.00      | 1.00   | 1.00     | 135,444 |

* **Correlation (ingredient concentration vs cold-weather preference):** `0.2826`

---

## Key Takeaways

* Cold-weather buyers prefer **higher concentrations of humectants and occlusives**
* Ingredient optimization according to **seasonal skincare needs** increases product relevance and sales potential
* High accuracy and recall affirm the model's effectiveness in aligning product formulation with consumer behavior

---

## Roles & Responsibilities

* **Spearheaded** a predictive recommendation model using `CatBoostClassifier` to classify customers based on key ingredient concentration matching by identifying low-performing sales revenue product **P442990** and other products during cold months (**January–April**).

  * Target labels: `1` (above median concentration match of `0.075`) and `0` (below).
  * Achieved:

    * **99.93% Accuracy**
    * **99.95% F1 Score**
    * **99.89% Precision**
    * **100% Recall**
    * Trained on over **8k products** and **1 million reviews**.
* **Refined feature engineering** integrating:

  * Seasonal skincare trends
  * Customer purchase history
  * Chemical ingredient composition
    Optimized product formulation strategies and uncovered a **moderate correlation (0.2826)** between ingredient concentration and cold-weather skincare preferences (2019–2023).
* **Designed visual analytics tools** to examine:

  * Ingredient effectiveness
  * Customer segmentation
  * Demand fluctuations
    Provided **actionable insights** for targeted marketing and product positioning in **winter skincare campaigns**.
    
---

## 🎥 Resources

* 🔗 [**Between the Lines Study Jam Recording**](https://lnkd.in/gvnQHNMH)
* 🔗 [**Between the Lines Model**](https://lnkd.in/gbVS5ETw)
* 🔗 [**Between the Lines AI/ML API**](https://lnkd.in/g3nWF8j6)
