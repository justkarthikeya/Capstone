from django import forms


class PredictForm(forms.Form):
    # Basic demographic/account fields
    gender = forms.ChoiceField(choices=[('Female','Female'),('Male','Male')], initial='Female')
    SeniorCitizen = forms.ChoiceField(choices=[('No','No'),('Yes','Yes')], initial='No')
    Partner = forms.ChoiceField(choices=[('No','No'),('Yes','Yes')], initial='Yes')
    Dependents = forms.ChoiceField(choices=[('No','No'),('Yes','Yes')], initial='No')

    tenure = forms.IntegerField(min_value=0, max_value=200, initial=1)
    Contract = forms.ChoiceField(choices=[('Month-to-month','Month-to-month'),('One year','One year'),('Two year','Two year')], initial='Month-to-month')

    PhoneService = forms.ChoiceField(choices=[('No','No'),('Yes','Yes')], initial='Yes')
    MultipleLines = forms.ChoiceField(choices=[('No','No'),('Yes','Yes'),('No phone service','No phone service')], initial='No')

    InternetService = forms.ChoiceField(choices=[('DSL','DSL'),('Fiber optic','Fiber optic'),('No','No')], initial='DSL')
    OnlineSecurity = forms.ChoiceField(choices=[('No','No'),('Yes','Yes'),('No internet service','No internet service')], initial='No')
    OnlineBackup = forms.ChoiceField(choices=[('No','No'),('Yes','Yes'),('No internet service','No internet service')], initial='Yes')
    DeviceProtection = forms.ChoiceField(choices=[('No','No'),('Yes','Yes'),('No internet service','No internet service')], initial='No')
    TechSupport = forms.ChoiceField(choices=[('No','No'),('Yes','Yes'),('No internet service','No internet service')], initial='No')
    StreamingTV = forms.ChoiceField(choices=[('No','No'),('Yes','Yes'),('No internet service','No internet service')], initial='No')
    StreamingMovies = forms.ChoiceField(choices=[('No','No'),('Yes','Yes'),('No internet service','No internet service')], initial='No')

    MonthlyCharges = forms.DecimalField(min_value=0, max_value=10000, decimal_places=2, initial=29.85)
    TotalCharges = forms.DecimalField(min_value=0, max_value=100000, decimal_places=2, initial=29.85)

    PaymentMethod = forms.ChoiceField(choices=[('Electronic check','Electronic check'),('Mailed check','Mailed check'),('Bank transfer (automatic)','Bank transfer (automatic)'),('Credit card (automatic)','Credit card (automatic)')], initial='Electronic check')
    PaperlessBilling = forms.ChoiceField(choices=[('No','No'),('Yes','Yes')], initial='Yes')
