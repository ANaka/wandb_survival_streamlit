
import streamlit as st
import cloudpathlib
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import lifelines
import altair as alt

@st.cache(hash_funcs={lifelines.fitters.coxph_fitter.CoxPHFitter: lambda _: None})
def load_model(
    s3_uri='s3://loyal-public-bucket-not-secure/20220511_streamlit_survival_model.pkl',
    ):
    fp = cloudpathlib.CloudPath(s3_uri)
    with open(fp, "rb") as f:
        return pickle.load(f)

model = load_model()
age_years = st.slider(label='age', min_value=0., max_value=20., value=0.)
weight_lbs = st.slider(label='weight', min_value=5, max_value=200, value=5)
sex = st.selectbox('Sex', ['Male', 'Female'])
is_male = (sex=='Male')  # patsy being sexist smh
desexed = st.selectbox('Spayed/Neutered', ['True', 'False'])
is_desexed = (desexed == 'True')

def format_model_inputs(
    age_years:float,
    is_male: bool = False,
    is_desexed: bool = True,
    weight_lbs:float = 50.,
    )-> pd.DataFrame:
    return pd.DataFrame({
            'dead': False,
            'age_years': age_years,
            'sex[T.M]': is_male,
            'is_desexed[T.True]': is_desexed, 
            'weight_lbs': weight_lbs}, 
            index=[0])

times = np.arange(0, 22, 0.5)
df = format_model_inputs(age_years=age_years, weight_lbs=weight_lbs, is_male=is_male, is_desexed=is_desexed)
adjusted_times = times + age_years
S = model.predict_survival_function(df=df, times=times, conditional_after=age_years).iloc[:, 0]
S.index = adjusted_times

# f,ax = plt.subplots(figsize=(10,5))
# ax.plot(S)
# ax.set_xlim(left=0, right=22)
# st.pyplot(f)

# st.line_chart(data=S)

S = S.reset_index().rename(columns={'index': 't', 0:'S'}).query('t < 22.')
c = alt.Chart(S).mark_line().encode(
    x=alt.X('t',scale=alt.Scale(domain=[0, 22.])),
    y='S'
)
st.altair_chart(c, use_container_width=True)