import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('pricing_optimization_model.pkl')

# Feature names expected by the model
feature_names = [
    'Year', 'Month', 'Day', 'Demand_Elasticity', 'Warehouse_Whse_J', 'Warehouse_Whse_S',
    'Product_Category_Category_003', 'Product_Category_Category_004', 'Product_Category_Category_005',
    'Product_Category_Category_006', 'Product_Category_Category_007', 'Product_Category_Category_008',
    'Product_Category_Category_009', 'Product_Category_Category_011', 'Product_Category_Category_013',
    'Product_Category_Category_015', 'Product_Category_Category_017', 'Product_Category_Category_018',
    'Product_Category_Category_019', 'Product_Category_Category_020', 'Product_Category_Category_021',
    'Product_Category_Category_022', 'Product_Category_Category_023', 'Product_Category_Category_024',
    'Product_Category_Category_025', 'Product_Category_Category_026', 'Product_Category_Category_028',
    'Product_Category_Category_030', 'Product_Category_Category_031', 'Product_Category_Category_032',
    'Product_Category_Category_033'
]


# Define the function that makes predictions
def predict_order_demand(Year, Month, Day, Demand_Elasticity, Warehouse_Whse_J, Warehouse_Whse_S, *Product_Categories):
    # Convert the product categories into a list and ensure it has the same length as the number of categories
    if len(Product_Categories) != len(feature_names) - 6:  # excluding Year, Month, Day, Elasticity, Warehouse columns
        return "Error: Incorrect number of product category inputs."

    # Combine the inputs into a single list
    input_data = [Year, Month, Day, Demand_Elasticity, Warehouse_Whse_J, Warehouse_Whse_S] + list(Product_Categories)

    # Create a dataframe with the same columns as the model expects
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Make the prediction
    prediction = model.predict(input_df)

    return prediction[0]


# Create the Gradio interface
with gr.Blocks() as demo:
    # Define input components using gr.Row and gr.Column for organization
    with gr.Row():
        year_input = gr.Slider(minimum=2010, maximum=2025, value=2022, label="Year")
        month_input = gr.Slider(minimum=1, maximum=12, value=6, label="Month")
        day_input = gr.Slider(minimum=1, maximum=31, value=15, label="Day")

    with gr.Row():
        demand_elasticity_input = gr.Number(value=0.1, label="Demand Elasticity")
        warehouse_whse_j_input = gr.Checkbox(value=True, label="Warehouse Whse_J")
        warehouse_whse_s_input = gr.Checkbox(value=False, label="Warehouse Whse_S")

    # Input features for product categories (25 sliders instead of 31)
    category_inputs = [
        gr.Slider(minimum=0, maximum=1, value=0, label=f"Product Category {i:03d}")
        for i in range(3, 34)
    ]
    category_inputs = category_inputs[:25]  # Only take the first 25 categories

    # Button to trigger prediction
    predict_button = gr.Button("Predict Order Demand")

    # Output textbox for the prediction result
    output = gr.Textbox(label="Predicted Order Demand")

    # Define interaction between inputs and output
    predict_button.click(
        predict_order_demand,
        inputs=[year_input, month_input, day_input, demand_elasticity_input, warehouse_whse_j_input,
                warehouse_whse_s_input] + category_inputs,
        outputs=output
    )

# Launch the Gradio interface
demo.launch()
