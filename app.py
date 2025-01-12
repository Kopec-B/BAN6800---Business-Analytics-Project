import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('pricing_optimization_model.pkl')


# Function to predict order demand
def predict_order_demand(year, month, day, demand_elasticity, warehouse_whse_j, warehouse_whse_s, category_003,
                         category_004, category_005, category_006, category_007, category_008, category_009,
                         category_011, category_013, category_015, category_017, category_018, category_019,
                         category_020, category_021, category_022, category_023, category_024, category_025,
                         category_026, category_028, category_030, category_031, category_032, category_033):
    # Prepare the input data as a pandas DataFrame
    input_data = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Demand_Elasticity': [demand_elasticity],
        'Warehouse_Whse_J': [warehouse_whse_j],
        'Warehouse_Whse_S': [warehouse_whse_s],
        'Product_Category_Category_003': [category_003],
        'Product_Category_Category_004': [category_004],
        'Product_Category_Category_005': [category_005],
        'Product_Category_Category_006': [category_006],
        'Product_Category_Category_007': [category_007],
        'Product_Category_Category_008': [category_008],
        'Product_Category_Category_009': [category_009],
        'Product_Category_Category_011': [category_011],
        'Product_Category_Category_013': [category_013],
        'Product_Category_Category_015': [category_015],
        'Product_Category_Category_017': [category_017],
        'Product_Category_Category_018': [category_018],
        'Product_Category_Category_019': [category_019],
        'Product_Category_Category_020': [category_020],
        'Product_Category_Category_021': [category_021],
        'Product_Category_Category_022': [category_022],
        'Product_Category_Category_023': [category_023],
        'Product_Category_Category_024': [category_024],
        'Product_Category_Category_025': [category_025],
        'Product_Category_Category_026': [category_026],
        'Product_Category_Category_028': [category_028],
        'Product_Category_Category_030': [category_030],
        'Product_Category_Category_031': [category_031],
        'Product_Category_Category_032': [category_032],
        'Product_Category_Category_033': [category_033]
    })

    # Use the model to predict the order demand
    prediction = model.predict(input_data)

    return prediction[0]


# Define the Gradio interface
iface = gr.Interface(
    fn=predict_order_demand,
    inputs=[
        gr.Slider(minimum=2012, maximum=2025, step=1, label="Year", value=2022),
        gr.Slider(minimum=1, maximum=12, step=1, label="Month", value=6),
        gr.Slider(minimum=1, maximum=31, step=1, label="Day", value=15),
        gr.Slider(minimum=-1.0, maximum=1.0, step=0.1, label="Demand Elasticity", value=0.05),
        gr.Slider(minimum=0, maximum=1, step=1, label="Warehouse Whse_J", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Warehouse Whse_S", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 003", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 004", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 005", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 006", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 007", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 008", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 009", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 011", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 013", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 015", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 017", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 018", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 019", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 020", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 021", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 022", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 023", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 024", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 025", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 026", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 028", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 030", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 031", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 032", value=0),
        gr.Slider(minimum=0, maximum=1, step=1, label="Category 033", value=0)
    ],
    outputs=gr.Number(label="Predicted Order Demand")
)

# Launch the Gradio app with sharing enabled
iface.launch(share=True)
