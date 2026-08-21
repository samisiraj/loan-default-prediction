import gradio as gr
import requests

API_URL = 'http://127.0.0.1:8080/predict'

def predict_loan(
    loan_limit,
    gender,
    approv_in_adv,
    loan_type,
    loan_purpose,
    credit_worthiness,
    open_credit,
    business_or_commercial,
    loan_amount,
    term,
    neg_ammortization,
    interest_only,
    lump_sum_payment,
    property_value,
    construction_type,
    occupancy_type,
    secured_by,
    total_units,
    income,
    credit_score,
    co_applicant_credit_type,
    age,
    submission_of_application,
    ltv,
    region,
    security_type,
    dtir1,
):
    payload = {
        "loan_limit": loan_limit,
        "gender": gender,
        "approv_in_adv": approv_in_adv,
        "loan_type": loan_type,
        "loan_purpose": loan_purpose,
        "credit_worthiness": credit_worthiness,
        "open_credit": open_credit,
        "business_or_commercial": business_or_commercial,
        "loan_amount": loan_amount,
        "term": term,
        "neg_ammortization": neg_ammortization,
        "interest_only": interest_only,
        "lump_sum_payment": lump_sum_payment,
        "property_value": property_value,
        "construction_type": construction_type,
        "occupancy_type": occupancy_type,
        "secured_by": secured_by,
        "total_units": total_units,
        "income": income,
        "credit_score": credit_score,
        "co-applicant_credit_type": co_applicant_credit_type,
        "age": age,
        "submission_of_application": submission_of_application,
        "ltv": ltv,
        "region": region,
        "security_type": security_type,
        "dtir1": dtir1,
    }
    
    try:
        response = requests.post(url=API_URL, json=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        detail = response.json().get("detail", response.text)
        return None, None, f"Input error: {detail}"
    except requests.exceptions.RequestException as e:
        return None, None, f"Could not reach prediction server: {e}"
    
    result = response.json()
    default_probability = result["default_probability"]
    prediction = result['prediction']

    if prediction:
        status = 'Likely to Default'
    else:
        status = 'Unlikely to Default'
        
    return(
        default_probability,
        prediction,
        status
    )
    
    
with gr.Blocks(title="Loan Default Predictor") as demo:

    gr.Markdown(
        """
        #  Loan Default Predictor

        Enter the applicant and loan information.
        """
    )
    
#--ApplicantInfo------
    gr.Markdown("## Applicant Information")

    with gr.Row():

        gender = gr.Dropdown(
            choices=[
                ('Male', 'male'),
                ('Female', 'female'),
                ('Joint', 'joint'),
                ('Unavailable', 'sex_not_available')
            ],
            label="Gender",
            value="male",
            info="Select Applicant\'s Gender"
        )

        age = gr.Dropdown(
            choices=[
                "<25",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65-74",
                ">74"
            ],
            label="Age",
            value="25-34",
            info='Select Applicant\'s Age Group'
        )

        income = gr.Number(
            label='Enter Annual Income in USD (Optional)',
            minimum=0,
            maximum=578580,
            info='(Dataset source does not specify the currency for income)'
        )

    with gr.Row():

        credit_score = gr.Number(
            label="Credit Score",
            minimum=500,
            maximum=900,
            info='Enter Credit Score between 500-900',
            value=650
        )

        co_applicant_credit_type = gr.Dropdown(
            choices=[
                ('CIBIL', 'cib'),
                ('Experian', 'exp')
            ],
            label="Co-applicant Credit Type",
            value="cib"
        )

#--LoanInfo---------------
    gr.Markdown("## Loan Information")

    with gr.Row():

        loan_limit = gr.Dropdown(
            choices=[
                ('Conforming','cf'),
                ('Non-Conforming', 'ncf')
            ],
            label="Loan Limit",
            value="cf"
        )
        
        approv_in_adv = gr.Dropdown(
            choices =[
                ('Pre-Approved Loan', 'pre'),
                ('Regular Loan', 'nopre')
            ],
            label='Approval in Advance',
            value='pre'
        )
       
        loan_type = gr.Dropdown(
            choices=[
                ("Type 1", "type1"),
                ("Type 2", "type2"),
                ("Type 3", "type3"),
            ],
            label='Loan Type',
            info="Dataset-defined loan type. The source does not provide a clear description for each type.",
            value='type1'
        )
        
    with gr.Row():
        
        loan_purpose = gr.Dropdown(
            choices=['p1', 'p2', 'p3', 'p4'],
            label='Loan Purpose',
            info='Dataset-defined codes. The source does not provide clear descriptions for p1–p4.',
            value="p1"
        )
        
        credit_worthiness = gr.Dropdown(
            choices=[
                'l1',
                'l2'
            ],
            label='Credit Worthiness',
            info='Dataset-defined codes. The source does not provide a clear description for each type.',
            value='l1'
        )
        
        open_credit = gr.Dropdown(
            choices=[
                ("No Open Credit", "nopc"),
                ("Open Credit", "opc"),
            ],
            label="Open Credit",
            value="nopc",
        )
        
    with gr.Row():
        
        business_or_commercial = gr.Dropdown(
            choices=[
                ('No', 'nob/c'),
                ('Yes', 'b/c')
            ],
            label='Business or Personal',
            info='Choose if loan is for Business/Commercial purposes (YES) or Personal use (NO)',
            value='nob/c'
        )
        
        loan_amount = gr.Number(
            label='Enter Loan Amount in USD from 16500',
            info='(Dataset source does not specify the currency for income)',
            minimum=16500,
            maximum=3576500,
            value=100000
        )
        
        term = gr.Number(
            label='Enter Loan Tenure in Months from 12 to 360',
            minimum=12,
            maximum=360,
            value=180
        )
        
    with gr.Row():
        
        neg_ammortization = gr.Dropdown(
            choices=[
                ('Yes', 'neg_amm'),
                ('No', 'not_neg')
            ],
            label='Negative Ammortization',
            info='Choose YES if loan allows for Negative Ammortization',
            value='not_neg'
        )
        
        interest_only = gr.Dropdown(
            choices=[
                ('Yes', 'int_only'),
                ('No', 'not_int')
            ],
            label='Interest Only Option',
            info='Choose YES if Loan Has an Interest-Only Payment Option',
            value='not_int'
        )
        
        lump_sum_payment = gr.Dropdown(
            choices=[
                ("Yes", "lpsm"),
                ("No", "not_lpsm")
            ],
            label="Lump Sum Payment",
            info="Choose YES if the loan has a lump-sum payment option.",
            value="not_lpsm"
        )

        submission_of_application = gr.Dropdown(
            choices=[
                ("To Institution", "to_inst"),
                ("Not to Institution", "not_inst")
            ],
            label="Submission of Application",
            info="Dataset-defined application submission category.",
            value="to_inst"
        )


#--PropertyInfo------
    gr.Markdown("## Property Information")

    with gr.Row():

        property_value = gr.Number(
            label="Property Value (Optional)",
            placeholder="Leave blank if unavailable",
            minimum=8000,
            maximum=16508000,
            info="Enter the property's value if available."
        )

        construction_type = gr.Dropdown(
            choices=[
                ("Site Built", "sb"),
                ("Manufactured Home", "mh")
            ],
            label="Construction Type",
            info="Select the property's construction type.",
            value="sb"
        )

        occupancy_type = gr.Dropdown(
            choices=[
                ("Primary Residence", "pr"),
                ("Investment Property", "ir"),
                ("Second Residence", "sr")
            ],
            label="Occupancy Type",
            value="pr"
        )

    with gr.Row():

        secured_by = gr.Dropdown(
            choices=[
                ("Home", "home"),
                ("Land", "land")
            ],
            label="Secured By",
            value="home"
        )

        total_units = gr.Dropdown(
            choices=[
                ("1 Unit", "1u"),
                ("2 Units", "2u"),
                ("3 Units", "3u"),
                ("4 Units", "4u")
            ],
            label="Total Units",
            info='Number of Units in the Property Being Financed',
            value="1u"
        )

        region = gr.Dropdown(
            choices=[
                ('North','north'),
                ('South', 'south'),
                ('Central', 'central'),
                ('North-East', 'north-east')
            ],
            label='Region',
            info='Geographic region as defined by the dataset.',
            value='north'
        )

    security_type = gr.Dropdown(
        choices=[
            ('Direct', 'direct'),
            ('Indirect', 'indriect')
        ],
        label='Security Type',
        info='Type of security/collateral backing the loan.',
        value="direct"
    )


#--Additional Loan Metrics------
    gr.Markdown("## Additional Loan Metrics")

    with gr.Row():

        ltv = gr.Number(
            label="Loan-to-Property Value Ratio (LTV) (Optional)",
            placeholder="Leave blank if unavailable",
            minimum=0.96,
            maximum=7831.25,
            info="Optional. Leave blank if unavailable. Accepted range: 0.96–7831.25"
        )

        dtir1 = gr.Number(
            label="Debt-to-Income Ratio (DTIR) (Optional)",
            placeholder="Leave blank if unavailable",
            minimum=5,
            maximum=61,
            info="Optional. Leave blank if unavailable. Accepted range: 5–61"
        )


#--Prediction------
    gr.Markdown("## Prediction")

    predict_button = gr.Button(
        "Predict Loan Default",
        variant="primary"
    )

    with gr.Row():

        default_probability = gr.Number(
            label='Default Probability'
        )

        prediction = gr.Number(
            label='Prediction'
        )

        status = gr.Textbox(
            label='Result'
        )

    predict_button.click(
        fn=predict_loan,
        inputs=[
            loan_limit,
            gender,
            approv_in_adv,
            loan_type,
            loan_purpose,
            credit_worthiness,
            open_credit,
            business_or_commercial,
            loan_amount,
            term,
            neg_ammortization,
            interest_only,
            lump_sum_payment,
            property_value,
            construction_type,
            occupancy_type,
            secured_by,
            total_units,
            income,
            credit_score,
            co_applicant_credit_type,
            age,
            submission_of_application,
            ltv,
            region,
            security_type,
            dtir1,
        ],
        outputs=[
            default_probability,
            prediction,
            status
        ]
    )


if __name__ == "__main__":
    demo.launch()