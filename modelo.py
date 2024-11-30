# # 1. Configurar ambiente

#Instalando bibliotecas necesarias
import pandas as pd
import re
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load('es_core_news_md')
import jellyfish
import requests
import nltk
global diccionario_irregulares, documento, lista_frases, lista_frases_normalizadas
import warnings, os
warnings.filterwarnings('ignore')
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
from groq import Groq
import logging

ruta = os.path.dirname(__file__)

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
UMBRALES = os.getenv('UMBRALES')
# Configuraciones generales


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# #2. Tratamiento de datos
#Función para encontrar la raiz de las palabras
def raiz(palabra):
  radio=0
  palabra_encontrada=palabra
  for word in lista_verbos:
    confianza = jellyfish.jaro_similarity(palabra, word)
    if (confianza>=0.93 and confianza>=radio):
      radio=confianza
      palabra_encontrada=word
  return palabra_encontrada

def tratamiento_texto(texto):
  trans = str.maketrans('áéíóú','aeiou')
  texto = texto.lower()
  texto = texto.translate(trans)
  texto = " ".join(texto.split())
  return texto

#Función para reemplazar el final de una palabra por 'r'
def reemplazar_terminacion(palabra):
  patron = r"(es|me|as|te|ste)$"
  nueva_palabra = re.sub(patron, "r", palabra)
  return nueva_palabra.split()[0]

#Función para adicionar o eliminar tokens
def revisar_tokens(texto, tokens):
  if len(tokens)==0:
    if [x for x in ['Geo Perú', 'geoperu', 'GeoPeru', 'geo peru'] if x in tratamiento_texto(texto)]: tokens.append('GeoPerú')
    elif [x for x in ['geo referencia', 'georreferencial'] if x in tratamiento_texto(texto)]: tokens.append('georreferencias')
  else:
    elementos_a_eliminar = ["cual", "que", "quien", "cuanto", "cuando", "como"]
    if 'hablame' in texto and 'hablar' in tokens: tokens.remove('hablar')
    elif 'cuentame' in texto and 'contar' in tokens: tokens.remove('contar') 
    elif 'hago' in texto and 'hacer' in tokens: tokens.remove('hacer') 
    elif 'entiendes' in texto and 'entender' in tokens: tokens.remove('entender') 
    elif 'sabes' in texto and 'saber' in tokens: tokens.remove('saber') 
    tokens = [x.replace('datar','data').replace('datos','dato') for x in tokens if x not in elementos_a_eliminar]
  return tokens

#Función para devolver los tokens normalizados del texto
def normalizar(texto):
  try:
    tokens=[]
    tokens=revisar_tokens(texto, tokens)
    if 'GeoPerú' in tokens:
      texto = ' '.join(texto.split()[:15])
    else:
      texto = ' '.join(texto.split()[:25])

    doc = nlp(texto)
    for t in doc:
      lemma=diccionario_irregulares.get(t.text, t.lemma_.split()[0])
      lemma=re.sub(r'[^\w\s+\-*/]', '', lemma)
      if t.pos_ in ('VERB','PROPN','PRON','NOUN','AUX','SCONJ','ADJ','ADV','NUM') or lemma in lista_verbos:
        if t.pos_=='VERB':
          lemma = reemplazar_terminacion(lemma)
          tokens.append(raiz(tratamiento_texto(lemma)))
        else:
          tokens.append(tratamiento_texto(lemma))
    tokens = list(dict.fromkeys(tokens))
    tokens = list(filter(None, tokens))
    tokens = revisar_tokens(texto, tokens)
    return tokens
  except Exception as e:
    logger.error(f"Error al normalizar texto: {str(e)}")
    raise ChatbotError(f"Error en normalización: {str(e)}")

#Función normalizar que se utilizó para entrenar el modelo
def normalizar_modelo(texto):
  doc = nlp(texto)
  tokens=[]
  if len(doc)<=3:
    for t in doc:
      if t.pos_=='VERB':
        tokens.append(raiz(t.lemma_))
      else:
        tokens.append(t.lemma_)
  else:
    for t in doc:
      if (t.pos_ in ('VERB','PROPN','PRON','NOUN','AUX','SCONJ','DET','ADJ','ADV') or any(t.dep_.startswith(elemento) for elemento in ['ROOT'])):
        if t.pos_=='VERB':
          tokens.append(raiz(t.lemma_))
        else:
          tokens.append(t.lemma_)
  tokens = list(dict.fromkeys(tokens))
  tokens = tokens[:10]
  tokens = ' '.join(tokens)
  return tratamiento_texto(tokens)

# # 3. Cargar bases de verbos
#Importando verbos en español

trans = str.maketrans('áéíóú','aeiou')
ruta_archivo = os.path.join(ruta, 'verbos.txt')
lista_verbos = []
with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
  contenido = archivo.read().strip()
lineas = contenido.split('\n')
for linea in lineas:
  palabra = linea.strip()
  lista_verbos.append(palabra)
nuevos_verbos = ['costar', 'referir', 'datar']
lista_verbos.extend(nuevos_verbos)
lista_verbos=list(set(lista_verbos))

# Definir una lista de verbos irregulares y sus conjugaciones en pasado, presente, futuro, imperfecto, pretérito y condicional
verbos_irregulares = [
    ('ser', 'soy', 'eres', 'seras', 'eras', 'es', 'serias'),
    ('estar', 'estuviste', 'estas', 'estaras', 'estabas', 'estuviste', 'estarias'),
    ('ir', 'fuiste', 'vas', 'iras', 'ibas', 'fuiste', 'irias'),
    ('ir', 'fuiste', 'vaya', 'iras', 'ibas', 'fuiste', 'irias'),
    ('tener', 'tuviste', 'tienes', 'tendras', 'tenias', 'tuviste', 'tendrias'),
    ('hacer', 'hiciste', 'haces', 'haras', 'hacias', 'hiciste', 'harias'),
    ('decir', 'dijiste', 'dices', 'diras', 'decias', 'dijiste', 'dirias'),
    ('decir', 'dimar', 'dime', 'digame', 'dimir', 'dimo', 'dimiria'),
    ('poder', 'pudiste', 'puedes', 'podras', 'podias', 'pudiste', 'podrias'),
    ('saber', 'supiste', 'sabes', 'sabras', 'sabias', 'supiste', 'sabrias'),
    ('poner', 'pusiste', 'pones', 'pondras', 'ponias', 'pusiste', 'pondrias'),
    ('ver', 'viste', 'ves', 'veras', 'veias', 'viste', 'verias'),
    ('dar', 'diste', 'das', 'daras', 'dabas', 'diste', 'darias'),
    ('dar', 'damar', 'dame', 'daras', 'dabas', 'darme', 'darias'),
    ('venir', 'viniste', 'vienes', 'vendras', 'venias', 'viniste', 'vendrias'),
    ('haber', 'haya', 'has', 'habras', 'habias', 'hubiste', 'habrias'),
    ('caber', 'cupiste', 'cabes', 'cabras', 'cabias', 'cupiste', 'cabrias'),
    ('valer', 'valiste', 'vales', 'valdras', 'valias', 'valiste', 'valdrias'),
    ('querer', 'quisiste', 'quieres', 'querras', 'querias', 'quisiste', 'querrias'),
    ('llegar', 'llegaste', 'llegares', 'llegaras', 'llegarias', 'llegaste', 'llegarrias'),
    ('hacer', 'hiciste', 'haces', 'haras', 'hacias', 'hiciste', 'harias'),
    ('decir', 'dijiste', 'dices', 'diras', 'decias', 'dijiste', 'dirias'),
    ('poder', 'pudiste', 'puedes', 'podras', 'podias', 'pudiste', 'podria'),
    ('contar', 'contaste', 'cuentas', 'contaras', 'contabas', 'cuentame', 'contarias'),
    ('saber', 'supiste', 'sabes', 'sabras', 'sabias', 'supiste', 'sabrias'),
    ('costar', 'cuesta', 'cuestan', 'costo', 'costaria', 'costarian', 'cuestas'),
    ('durar', 'duraste', 'duro', 'duraras', 'durabas', 'duraste', 'durarias')
]

# Crear el DataFrame
diccionario_irregulares = {}
df = pd.DataFrame(verbos_irregulares, columns=['Verbo', 'Pasado', 'Presente', 'Futuro', 'Imperfecto', 'Pretérito', 'Condicional'])
for columna in df.columns:
  if columna != 'Verbo':
    for valor in df[columna]:
      diccionario_irregulares[valor] = df.loc[df[columna] == valor, 'Verbo'].values[0]


# # 4. Cargar bases de documentos
#Importando bases de dialogo fluído
txt_folder_path = os.path.join(ruta, 'txt')
lista_documentos=[x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
lista_dialogos, lista_dialogos_respuesta, lista_tipo_dialogo = [],[],[]
for idx in range(len(lista_documentos)):
  f=open(txt_folder_path+'/'+lista_documentos[idx], 'r', encoding='utf-8', errors='ignore')
  flag,posicion = True,0
  for line in f.read().split('\n'):
    if flag:
      line = tratamiento_texto(line)
      line = re.sub(r"[^\w\s]", '', line)
      lista_dialogos.append(line)
      lista_tipo_dialogo.append(lista_documentos[idx].replace('.txt', ''))
    else:
      lista_dialogos_respuesta.append(line)
      posicion+=1
    flag=not flag

#Creando Dataframe de diálogos
datos = {'dialogo':lista_dialogos,'respuesta':lista_dialogos_respuesta,'tipo':lista_tipo_dialogo,'interseccion':0,'similarity':0,'jaro_winkler':0,'probabilidad':0}
df_dialogo = pd.DataFrame(datos)
df_dialogo = df_dialogo.drop_duplicates(keep='first')
df_dialogo.reset_index(drop=True, inplace=True)

#Importando bases txt
txt_folder_path = os.path.join(ruta, 'dialogo')
lista_documentos=[x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
documento_txt = ''
for idx in range(len(lista_documentos)):
  with open(txt_folder_path+'/'+lista_documentos[idx], "r", encoding="utf-8") as archivo_txt:
    lector_txt = archivo_txt.read()
    for fila in lector_txt:
      documento_txt += fila

documento = documento_txt
lista_frases = nltk.sent_tokenize(documento,'spanish')
lista_frases_normalizadas = [' '.join(normalizar(x)) for x in lista_frases]

# # 5. Buscar respuesta del Chatbot
#Cargar el modelo entrenado
ruta_modelo = os.path.join(ruta, 'modelo')
Modelo_TF = BertForSequenceClassification.from_pretrained(ruta_modelo)
tokenizer_TF = BertTokenizer.from_pretrained(ruta_modelo)

#Función para verificar si el usuário inició un diálogo
def dialogo(user_response):
  user_response = tratamiento_texto(user_response)
  user_response = re.sub(r"[^\w\s]", '', user_response)
  df = df_dialogo.copy()
  vectorizer = TfidfVectorizer()
  dialogos_numero = vectorizer.fit_transform(df_dialogo['dialogo'])
  respuesta_numero = vectorizer.transform([user_response])
  for idx,row in df.iterrows():
    df.at[idx,'interseccion'] = len(set(user_response.split()) & set(row['dialogo'].split()))/len(user_response.split())
    df.at[idx,'similarity'] = cosine_similarity(dialogos_numero[idx], respuesta_numero)[0][0]
    df.at[idx,'jaro_winkler'] = jellyfish.jaro_similarity(user_response,row['dialogo'])
    df.at[idx,'probabilidad'] = max(df.at[idx,'interseccion'],df.at[idx,'similarity'],df.at[idx,'jaro_winkler'])
  df.sort_values(by=['probabilidad','jaro_winkler'], inplace=True, ascending=False)
  probabilidad = df['probabilidad'].head(1).values[0]
  tipo = df['tipo'].head(1).values[0]
  if probabilidad >= UMBRALES['similitud_dialogo']:
    logger.info(f'Respuesta encontrada por diálogo - Probabilidad: {probabilidad}')
    return df['respuesta'].head(1).values[0]
  return ''


#Función para dialogar utilizando el modelo Transformers
def clasificacion_modelo(pregunta):
  pregunta = re.sub(r"[^\w\s]", '', pregunta)
  frase = normalizar_modelo(pregunta)
  tokens = tokenizer_TF.encode_plus(
      frase,
      add_special_tokens=True,
      max_length=128,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
  )
  input_ids = tokens['input_ids']
  attention_mask = tokens['attention_mask']

  with torch.no_grad():
      outputs = Modelo_TF(input_ids, attention_mask)

  etiquetas_predichas = torch.argmax(outputs.logits, dim=1)
  etiquetas_decodificadas = etiquetas_predichas.tolist()
  #Se debe agregar la nueva etiqueta en caso se haya entrenado con un archivo distinto.
  diccionario = {0: 'Agradecimiento', 1: 'Aprendizaje', 2: 'Contacto', 3: 'Continuacion', 4: 'Despedida', 5: 'Edad', 6: 'Error', 7: 'Funcion', 8: 'GeoPeru', 9: 'Identidad', 10: 'Nombre', 11: 'Origen', 12: 'Otros', 13: 'Qa', 14: 'Saludos', 15: 'Sentimiento', 16: 'Usuario'}
  llave_buscada = etiquetas_decodificadas[0]
  clase_encontrada = diccionario[llave_buscada]

  #Buscar respuesta más parecida en la clase encontrada
  df = df_dialogo[df_dialogo['tipo'] == clase_encontrada]
  df.reset_index(inplace=True)
  vectorizer = TfidfVectorizer()
  dialogos_num = vectorizer.fit_transform(df['dialogo'])
  pregunta_num = vectorizer.transform([tratamiento_texto(pregunta)])
  similarity_scores = cosine_similarity(dialogos_num, pregunta_num)
  indice_pregunta_proxima = similarity_scores.argmax()
  if clase_encontrada not in ['Otros']:
    respuesta = df['respuesta'][indice_pregunta_proxima]
  else:
    respuesta = ''
  return respuesta
  
class GroqClient:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
    
    def obtener_respuesta(self, pregunta):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": pregunta}],
                model="llama3-8b-8192",
                stream=False
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error al llamar a Groq API: {str(e)}")
            return None

def respuesta_conAPi(pregunta):
    client = GroqClient(GROQ_API_KEY)
    return client.obtener_respuesta(pregunta)

#Función para devolver la respuesta de los documentos
def respuesta_documento(pregunta):
  pregunta = normalizar(pregunta)
  def contar_coincidencias(frase):
    return sum(1 for elemento in pregunta if elemento in frase) 

  diccionario = {valor: posicion for posicion, valor in enumerate(lista_frases_normalizadas)}
  lista = sorted(list(diccionario.keys()), key=contar_coincidencias, reverse=True)[:6]
  lista.append(' '.join(pregunta))
  TfidfVec = TfidfVectorizer(tokenizer=normalizar)
  tfidf = TfidfVec.fit_transform(lista)
  vals = cosine_similarity(tfidf[-1], tfidf)
  idx = vals.argsort()[0][-2]
  flat = vals.flatten()
  flat.sort()
  req_tfidf = round(flat[-2],2)
  if req_tfidf>=0.20:
    respuesta = lista_frases[diccionario[lista[idx]]]
  else:
    respuesta = ''
  return respuesta

class RespuestaHandler:
    def __init__(self, siguiente_handler=None):
        self.siguiente_handler = siguiente_handler
    
    def manejar(self, pregunta):
        respuesta = self.procesar(pregunta)
        if respuesta:
            return respuesta
        elif self.siguiente_handler:
            return self.siguiente_handler.manejar(pregunta)
        return 'Lo siento, no tengo respuesta para tu pregunta'
    
    def procesar(self, pregunta):
        raise NotImplementedError

class DialogoHandler(RespuestaHandler):
    def procesar(self, pregunta):
        return dialogo(pregunta)

class DocumentoHandler(RespuestaHandler):
    def procesar(self, pregunta):
        return respuesta_documento(pregunta)

class ModeloHandler(RespuestaHandler):
    def procesar(self, pregunta):
        return clasificacion_modelo(pregunta)

class APIHandler(RespuestaHandler):
    def procesar(self, pregunta):
        return respuesta_conAPi(pregunta)

def crear_cadena_respuesta():
    api_handler = APIHandler()
    modelo_handler = ModeloHandler(api_handler)
    documento_handler = DocumentoHandler(modelo_handler)
    dialogo_handler = DialogoHandler(documento_handler)
    return dialogo_handler

def respuesta_chatbot(pregunta):
    cadena = crear_cadena_respuesta()
    return cadena.manejar(pregunta)

class ChatbotError(Exception):
    pass