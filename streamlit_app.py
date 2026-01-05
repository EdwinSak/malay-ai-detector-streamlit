import streamlit as st
import joblib, time
from st_circular_progress import CircularProgress
st.title("Malay AI Text Detector", text_alignment='center', anchor=False)
import textwrap

COLUMN_HEIGHT = 725
PAGE_LAYOUT = 'wide'

st.set_page_config(layout=PAGE_LAYOUT)
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

cp = CircularProgress(
    label='AI Probability',
    value=50,
    key='circular_progress_ai_probs',
    color='green',
    size='Small',
)

def map_score(raw_score, threshold=0.5):
    if raw_score <= threshold:
        return (raw_score / threshold) * 0.5
    else:
        return 0.5 + (raw_score - threshold) / (1.0 - threshold) * 0.5

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


def generate_token_chart():
    if 'last_result' in st.session_state:
        models = ['mallam', 'electra', 'mistral', 'svm']
        tokens = [st.session_state.last_result['token_counts'][m] for m in models]
    else:
        tokens= [0,0,0,0]
    data = {
        'Model': ['Mallam', 'Electra', 'Mistral', 'SVM'],
        'Tokens': tokens
    }
    df = pd.DataFrame(data)

    fig = px.bar(
        df, 
        x='Model', 
        y='Tokens',
        color='Model',
        color_discrete_sequence=['#FFC759', '#FF7B9C', '#607196', '#BABFD1'],
        template='plotly_dark'
    )


    fig.update_traces(width=0.4)
    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        height=400,
        yaxis=dict(range=[0,None]),
        margin=dict(t=30, b=0, l=0, r=0),
        xaxis_title="", 
        showlegend=False,
    )
    
    return fig


def run_inference(input_text):
    try:
        import deployment_cloud
        print('submitted')
        st.session_state.analysis_running = True
        st.session_state.last_input = input_text
        # output = deployment_cloud.inference(mistral, mistral_tok, deployment_cloud.mistral_threshold, input_text)
        output = deployment_cloud.ensemble_inference(
            mallam=mallam,
            mallam_tok=mallam_tok,
            mistral=mistral,
            mistral_tok=mistral_tok,
            electra=electra,
            electra_tok=electra_tok,
            svm=svm,
            text=input_text
        )
        st.session_state.last_result = output
        print(output)
        st.toast(f'Inference succeeded! Text was written by {output["pred"]}', icon='✅')
        return output
    except Exception as e:
        st.toast('Inference failed! Please retry.', icon='❌')
        print("an error occured")
        raise e
    finally:
        st.session_state.analysis_running = False
        # st.rerun()


def generate_ai_score(name='Ensemble'):
    last_result = st.session_state.get('last_result', [])
    if last_result:
        print(last_result)
        probability = last_result['calibrated_probs'][name]
        print(f'probability: {probability}')
        score = map_score(probability, st.session_state.current_threshold)
        print(f'score: {score}')
        score = round(score*100, 2)
    else:
        score = 0
    remaining = 100 - score
    if score>=95:
        color = "#910808"
    elif score >= 90:
        color = "#E32A00"
        # color = '#f056a3'
    elif score >= 80:
        color = "#fc8916"
    elif score >= 60:
        color = "#e3ae10"
    elif score >= 40:
        color = '#56e1f0'
    elif score >= 30:
        color = "#b0e447"
    else:
        color = '#6df056'

    fig = go.Figure(data=[go.Pie(
        labels=['Probability', 'Remaining'],
        values=[score, remaining],
        hole=0.9,
        marker_colors = [color, '#f0f2f6'], 
        textinfo='none',       
        hoverinfo='label+percent',
        sort=False,         
    )])

    fig.update_layout(
        annotations=[dict(
            text=f"<b>{score}%</b>", 
            x=0.5, y=0.5, 
            font_size=30, 
            showarrow=False,
            font_color='#e8e8e8'
        )],
        showlegend=False,      
        margin=dict(t=0, b=0, l=0, r=0),
        height=130
    )
    return fig

def generate_ai_proportion():
    last_result = st.session_state.get('last_result', [])
    if last_result:
        print(last_result)
        sum = last_result['ai_sum']
        print(f'total_sum: {sum}')
    else:
        sum = 0 

    remaining = 5 - sum
    print(type(sum))
    print(type(remaining))


    fig = go.Figure(data=[go.Pie(
        labels=['AI', 'Human'],
        values=[sum, remaining],
        hole=0,
        marker=dict(
            colors=['#f06656', '#0e1117'],
            line=dict(color='#e8e8e8', width=1) 
            ), 
        textinfo='none',       
        hoverinfo='none',
        sort=False,        
    )])

    fig.update_layout(
        annotations=[dict(
            text=f"<b>{sum}/5<br>Models</b>", 
            x=0.5, y=0.5, 
            font_size=22, 
            showarrow=False,
            font_color='#e8e8e8'
        )],
        showlegend=False,      
        margin=dict(t=20, b=20, l=0, r=0),
        height=170
    )
    return fig

def generate_radar_chart():
    categories = ['Mallam', 'Mistral', 'Electra', 'SVM', 'Ensemble']
    if 'last_result' in st.session_state:
        calibrated_props = st.session_state.last_result['calibrated_probs']
        scores = [calibrated_props[m] *100 for m in categories]
    else:
        scores = [0,0,0,0,0]
    categories_closed = categories + [categories[0]]
    scores_closed = scores + [scores[0]]

    print(scores)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=categories_closed,
        fill='toself',
        name='AI Likelihood',
        line=dict(color="#923ADA"),
        marker=dict(size=8),
        mode='lines'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[50,50,50,50,50,50],
        theta=categories_closed,
        mode='lines',
        fill='none',
        line=dict(color='white', dash='dash', width=1),
        hoverinfo='none',
        name='Threshold'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,100],
                tickfont=dict(size=16, color='white'),
            ),
            angularaxis=dict(
                tickfont=dict(size=16, color='white', family="Arial"), # Font for Model Names
                rotation=90, # Starts Mallam at the top
                direction="clockwise"
            ),
            bgcolor='#0e1117',
            gridshape='linear'
        ),
        showlegend=False,
        margin=dict(t=30,b=20,l=40,r=40),
        height=300,
      
        paper_bgcolor='#0e1117'
    )

    return fig

import plotly.graph_objects as go


def generate_gauge_chart(name):
    last_result = st.session_state.get('last_result', [])
    if last_result:
        print(last_result)
        probability = last_result['calibrated_probs'][name]
        print(f'probability: {probability}')
        score = map_score(probability, st.session_state.current_threshold)
        print(f'score: {score}')
        score = round(score*100, 2)
    else:
        score = 0
    if score>=95:
        color = "#910808"
    elif score >= 90:
        color = "#E32A00"
        # color = '#f056a3'
    elif score >= 80:
        color = "#fc8916"
    elif score >= 60:
        color = "#e3ae10"
    elif score >= 50:
        color = '#56e1f0'
    elif score >= 30:
        color = "#b0e447"
    else:
        color = '#6df056'

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        number= {
            'font': {
                'size': 22
            },
            'suffix': '%'
        },
        domain = {'x': [0, 1], 'y': [0, 1]},
        # title = {'text': name, 'font': {'size': 16, 'color': 'white'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color':color}, 
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 0, 0, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(0, 0, 0, 0.2)'}, 
                {'range': [70, 100], 'color': 'rgba(0, 0, 0, 0.2)'}  
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor='#0e1117',
        font={'color': "white", 'family': "Arial"},
        height=100,
        margin=dict(t=10, b=20, l=0, r=0)
    )
    
    return fig


print('entire damn thing is rerun')
@st.fragment
def main_ui():
    @st.fragment
    def test():
        fig = generate_ai_score()
        fig2 = generate_ai_proportion()
        fig3 = generate_radar_chart()

        fig4 = generate_gauge_chart('Mallam')
        fig5 = generate_gauge_chart('Electra')
        fig6 = generate_gauge_chart('Mistral')
        fig7 = generate_gauge_chart('SVM')

        bar_chart = generate_token_chart()
         
        # fig4 = generate_ai_score('Mallam')
        # fig5 = generate_ai_score('Electra')
        # fig6 = generate_ai_score('Mistral')
        # fig7 = generate_ai_score('SVM')

        input_section, output_section= st.columns([2,1], gap='medium')
        dashboard_section = st.expander(label='Click to view additional analysis!')
        with input_section:
            with st.form("text_form", height=COLUMN_HEIGHT):
                input_text = st.text_area("Input text", height=COLUMN_HEIGHT-100, placeholder="Enter Malay text here...")
                submit = st.form_submit_button("Detect")
        with output_section:
            result_container = st.container(border=True, height=COLUMN_HEIGHT,gap='small')
            with result_container:
                st.subheader('Analysis', text_alignment='center', width='stretch', anchor=False)
                predictions_container = st.container(border=True, height='content', gap=None, vertical_alignment='center')
                percentage_container = st.container(border=True)
                model_proportions_container = st.container(border=True)
        with predictions_container:
            if 'last_result' in st.session_state:
                st.text("Your text was likely authored by", width='stretch', text_alignment='center')
                pred = st.session_state.last_result['pred']
                if pred == 'AI':
                    color = 'red'
                else:
                    color = 'rainbow'
                st.subheader(f':{color}[{pred}]', width='stretch', text_alignment='center', anchor=False)
            else:
                st.text("No text received.", width='stretch', text_alignment='center')

        with percentage_container:
            if 'last_result' in st.session_state:
                st.text("There's a", width='stretch', text_alignment='center')
                st.plotly_chart(fig, width='stretch')
                st.text(f'likelihood your text is from AI', width='stretch', text_alignment='center')
            else:
                st.text("AI Likelihood", width='stretch', text_alignment='center')
                st.plotly_chart(fig, width='stretch')
        with model_proportions_container:
            st.plotly_chart(fig2, width='stretch')
            st.text("thinks your text is AI-generated", width='stretch', text_alignment='center')
        with dashboard_section:
            radar_section, gauge_section = st.columns([2,1], border=True)
            # radar_section = st.container()
            with radar_section:
                st.subheader('Score Summary', anchor=False)
                with st.container(vertical_alignment='center', height='stretch'):
                    st.plotly_chart(fig3, width='stretch')
            # gauge_section = st.container()
            with gauge_section:
                st.subheader('Model''s Opinion', anchor=False)
                with st.container(gap='small'):
                    # st.subheader('MaLLaM', text_alignment='center', anchor=False)
                    st.text('MaLLaM', text_alignment='center')
                    st.plotly_chart(fig4, key='gauge_mallam', width='content')
                with st.container(gap='small'):
                    # st.subheader('Bahasa ELECTRA', text_alignment='center', anchor=False)
                    st.text('Bahasa ELECTRA', text_alignment='center')
                    st.plotly_chart(fig5, key='gauge_electra', width='content')
                with st.container(gap='small'):
                    # st.subheader('Malaysian Mistral', text_alignment='center', anchor=False)
                    st.text('Malaysian Mistral', text_alignment='center')
                    st.plotly_chart(fig6, key='gauge_mistral', width='content')
                with st.container(gap='small'):
                    # st.subheader('SVM', text_alignment='center', anchor=False)
                    st.text('SVM', text_alignment='center')
                    st.plotly_chart(fig7, key='gauge_svm', width='content')   
            with st.container(border=True):
                st.subheader('Token Count', anchor=False)
                st.plotly_chart(bar_chart, key='token_chart')

        if submit:
            if 'model_ready' not in st.session_state:
                st.toast('Model not ready yet, please wait!', icon='🛎️')
            elif ('last_input' in st.session_state) and st.session_state.last_input == input_text:
                st.warning('You are putting the same thing again. No evaluation was done.', icon="⚠️")
            else:
                st.toast('Running inference...', icon='🔍')
                st.session_state.analysis_running = True
                run_inference(input_text)
                st.rerun(scope='fragment')

    _, main_col, _ = st.columns([1,3,1])
    with main_col:
        disclaimer = st.expander(label='Disclaimer')
        with disclaimer:
            st.markdown(textwrap.dedent('''**Please read the following instructions carefully to ensure appropriate use of the Malay AI Detector.**
                \n- This detector is made for detecting **Malay** text only. Detection of any other languages including English are not supported as of right now.
                \n- Please ensure that there are atleast 100 words in the input for a more reliable estimate. Results with word count below 100 are not reliable and should not be taken as proof.   
                \n- The detector has a maximum limit of 512 tokens, which is roughly around 300-350 words. Exceeding the word limit would not result in an error, but the model will truncate the exceess tokens internally, making extra words effectively useless.
                \n- The results from this detector should only be used for reference and not as conclusive evidence of AI use. Please consult your respective AI ethics board for evaluation guidelines procedure.
                \n- This detector is by no means perfect. Paraphrasing and manual alteration of AI text may bypass detection. Therefore, do use other specialized tools to detect such attacks.         
                \n- Copying from AI directly for school assignments is strongly discouraged. Please use AI ethically and responsibly.
            '''))
        status = st.status("Initializing AI detector models...", expanded=True)
        test()
    
    return status



status = main_ui()
spinner_text = 'Initializing models and tokenizers, please wait...'
@st.cache_resource(
    show_spinner=False
)
def init_mallam():
    import deployment_cloud
    return deployment_cloud.initialize_mallam()

@st.cache_resource(
    show_spinner=False
)
def init_mistral():
    import deployment_cloud
    return deployment_cloud.initialize_mistral()

@st.cache_resource(
    show_spinner=False
)
def init_electra():
    import deployment_cloud
    return deployment_cloud.initialize_electra()

@st.cache_resource(
    show_spinner=False
)
def init_svm():
    import deployment_cloud
    return deployment_cloud.initialize_svm()


if "model_ready" not in st.session_state:
    with status:
        print('initializing models')
        st.write("Initializing inference engine...")
        import deployment_cloud

        st.write("Loading Mallam...")
        mallam, mallam_tok = init_mallam()
        
        st.write("Loading Mistral...")
        mistral, mistral_tok = init_mistral()
        
        st.write("Loading Electra...")
        electra, electra_tok = init_electra()

        st.write("Loading SVM...")
        svm = init_svm()
        
        st.session_state.model_ready = True
        st.session_state.current_threshold = deployment_cloud.ensemble_threshold
        print('current threshold  =' + str(deployment_cloud.ensemble_threshold))

        status.update(label="All models loaded!", state="complete", expanded=False)
        print('initalized models')
else:
    mallam, mallam_tok = init_mallam()
    mistral, mistral_tok = init_mistral()
    electra, electra_tok = init_electra()
    svm = init_svm()





if "model_ready" in st.session_state and 'ready_shown_notification' not in st.session_state:
    st.toast("Model Loaded Successfully!", icon="🚀")
    st.session_state.ready_shown_notification = True




# if st.session_state.analysis_running:
    


