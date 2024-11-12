import streamlit as st
import requests
import xml.etree.ElementTree as ET
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

# API 키
key = 'KPXkp9A5jCelRkUuOvT3EfPP0nOgvtQUiRySPCUmIPHtzYSE6G9/vLanfYwDPYz6JfL2K6HaXTIMOLyW1c3zwA=='

url_request = 'http://apis.data.go.kr/B551177/StatusOfPassengerWorldWeatherInfo/getPassengerArrivalsWorldWeather'

# API 요청 함수
def get_passenger_arrivals():
    url = url_request
    params = {
        'serviceKey': key, 
        'numOfRows': '10', 
        'pageNo': '1', 
        'from_time': '0000', 
        'to_time': '2400', 
        'airport': '', 
        'flight_id': '', 
        'airline': '', 
        'lang': 'K', 
        'type': 'xml'
    }
    response = requests.get(url, params=params)
    content = response.content.decode('utf-8')
    data = ET.fromstring(content)
    return data

def get_passenger_departures():
    url = url_request
    params = {
        'serviceKey': key, 
        'numOfRows': '10', 
        'pageNo': '1', 
        'from_time': '0000', 
        'to_time': '2400', 
        'airport': '', 
        'flight_id': '', 
        'airline': '', 
        'lang': 'K', 
        'type': 'xml'
    }
    response = requests.get(url, params=params)
    content = response.content.decode('utf-8')
    data = ET.fromstring(content)
    return data

def parse_weather_quality_data(data):
    items = data.find('.//items')
    weather_quality_info = []
    
    if items is not None:
        for item in items.findall('item'):
            flight_info = {
                '항공사': item.find('airline').text if item.find('airline') is not None else None,
                '편명': item.find('flightId').text if item.find('flightId') is not None else None,
                '예정시간': item.find('scheduleDateTime').text if item.find('scheduleDateTime') is not None else None,
                '변경시간': item.find('estimatedDateTime').text if item.find('estimatedDateTime') is not None else None,
                '출발공항': item.find('airport').text if item.find('airport') is not None else None,
                '현황': item.find('remark').text if item.find('remark') is not None else None,
                '공항코드': item.find('airportCode').text if item.find('airportCode') is not None else None,
                '날씨표출 요일': item.find('yoil').text if item.find('yoil') is not None else None,
                '습도': item.find('himidity').text if item.find('himidity') is not None else None,
                '풍속': item.find('wind').text if item.find('wind') is not None else None,
                '관측 기온': item.find('temp').text if item.find('temp') is not None else None,
                '체감 기온': item.find('senstemp').text if item.find('senstemp') is not None else None,
            }
            weather_quality_info.append(flight_info)
    else:
        st.write("정보가 존재하지 않습니다.")
    
    return weather_quality_info

# Streamlit UI 설정
st.title("공항 날씨 챗봇")
st.write("출발지 날씨에 대해 궁금한 점을 물어보세요!")
st.write("🙂 인천국제공항관련 비행기 시간정보도 알려드립니다.")

# 사용자 입력
user_input = st.text_input("질문을 입력하세요:", "")

if user_input:
    # 도착 및 출발 정보 호출
    arrivals_data = get_passenger_arrivals()
    departures_data = get_passenger_departures()
    
    # 날씨 정보 파싱
    arrivals_info = parse_weather_quality_data(arrivals_data)
    departures_info = parse_weather_quality_data(departures_data)

    # 날씨 정보 문서 생성
    documents = [Document(page_content=", ".join([f"{key}: {str(info[key])}" for key in ['항공사', '편명', '예정시간', '변경시간', '출발공항','습도', '풍속', '관측 기온', '체감 기온']])) for info in arrivals_info]
    documents += [Document(page_content=", ".join([f"{key}: {str(info[key])}" for key in ['항공사', '편명', '예정시간', '변경시간', '출발공항', '습도', '풍속', '관측 기온', '체감 기온']])) for info in departures_info]

    # 임베딩 함수 설정
    embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    db = FAISS.from_documents(documents, embedding_function)
    
    # 문서 검색
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5, 'fetch_k': 100})
    
    # 챗봇 응답 생성
    template = """
    반드시 사용자에게 한글로 알려주세요.
    너는 아주 친절한 공항 직원이야.
    사용자에게 친절한 말투로 잘 설명해줘.
    Answer the question as based only on the following context:
    {context}
     
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOllama(model="gemma2:9b", temperature=0, base_url="http://127.0.0.1:11434/")
    
    chain = RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x['question']),
        "question": lambda x: x['question']
    }) | prompt | llm
    
    response = chain.invoke({'question': user_input}).content
    
    st.markdown(response)