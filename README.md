**Low Performing Product in Sephora Dataset Product Recommendation Model**

**Data Set:** <a href="https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews" target="_blank">Sephora Products and Skincare Reviews</a>
  
**Objective:**  
Low Performing Product in Sephora Dataset Product Recommendation Model is a machine learning solution developed for the Between the Lines: Machine Learning Study Jam event on April 4, 2025. The model predicts customer purchasing behavior based on ingredient concentration matching with key chemical ingredients in low-performing sales revenue Product P442990 (REN Clean Skincare's Clean Screen Mineral SPF 30 Mattifying Face Sunscreen) and seasonal skincare needs.

**Key Insights:**  
Cold weather leads to increased transepidermal water loss (TEWL), requiring enhanced hydration and occlusive agents. Product P442990’s formulation integrates **emollients, humectants, antioxidants, and protective agents**, ensuring optimal skin protection during low temperatures and dry conditions.

### **Ingredient Analysis & Benefits**

1. **Humectants (e.g., Glycerin, Hyaluronic Acid, Panthenol)**  
   - Draw moisture into the skin, reducing dryness and flakiness.  
   - Maintain hydration levels by attracting water molecules from the environment and deeper skin layers.  
   - Reinforce the natural moisture barrier, preventing excessive water loss.

2. **Emollients (e.g., Squalane, Ceramides, Shea Butter)**  
   - Provide a soft, smooth texture to the skin by filling in micro-cracks.  
   - Restore the lipid barrier, ensuring long-lasting hydration.  
   - Help prevent irritation caused by cold-induced skin sensitivity.  

3. **Occlusive Agents (e.g., Dimethicone, Petrolatum, Beeswax)**  
   - Form a protective seal over the skin’s surface, preventing moisture loss.  
   - Shield the skin from external environmental stressors like wind and low humidity.  
   - Enhance the longevity of humectant and emollient action.  

4. **Antioxidants (e.g., Vitamin E, Green Tea Extract, Niacinamide)**  
   - Combat oxidative stress and inflammation exacerbated by harsh weather conditions.  
   - Strengthen skin resilience and repair damage caused by environmental aggressors.  
   - Enhance the overall effectiveness of moisturization by reducing free radical damage.  

5. **UV & Pollution Protectants (e.g., Zinc Oxide, Titanium Dioxide, Licorice Root Extract)**  
   - Provide an added layer of protection against UV exposure, which remains a risk in winter months.  
   - Reduce hyperpigmentation and irritation caused by cold-induced dryness.  
   - Support skin repair and minimize long-term damage from seasonal changes.  

### **Model-Based Analysis for Ingredient Optimization**  
Utilizing **CatBoostClassifier**, the model predicts the likelihood of customer preferences based on ingredient concentrations in P442990 (identified as the low-performing sales revenue) and other products during cold months (January-April). The model classifies customer data into two categories:
- **1:** Higher-than-median ingredient concentration match, indicating increased purchase likelihood (above calculated median concentration match of 0.075).
- **0:** Lower-than-median ingredient concentration match, indicating reduced purchase likelihood (below calculated median concentration match of 0.075).

#### **Classification Report:**  
- **Accuracy:** 0.9993  
- **F1 Score:** 0.9995  
- **Precision:** 0.9989  
- **Recall:** 1.0000  

| Category | Precision | Recall | F1-Score | Support |
|----------|------------|---------|-----------|----------|
| 0 (Low Match) | 1.00 | 1.00 | 1.00 | 61,020 |
| 1 (High Match) | 1.00 | 1.00 | 1.00 | 135,444 |

- **Correlation between cold-weather preferences and P442990’s ingredient concentration:** **0.2826**  

### **Key Takeaways from Model Analysis:**  
- Customers with preferences for **higher humectant and occlusive agent concentrations** are more likely to purchase P442990 during the winter months.
- The model confirms that optimizing ingredient ratios in line with **cold-weather skincare needs** leads to better customer targeting and improved formulation effectiveness.
- The high classification accuracy and recall demonstrate a strong predictive capability for matching ingredient concentration preferences with winter skincare needs.


