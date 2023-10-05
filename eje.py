
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import spacy
nlp = spacy.load('es_core_news_md')
import torch


ruta = os.path.dirname(__file__) 
# %% [markdown]
# #2. Tratamiento de datos

# %%
#Función para encontrar la raiz de las palabras escritas
def raiz(palabra):
  radio=0
  palabra_encontrada=palabra
  for word in lista_verbos:
    confianza = jellyfish.jaro_winkler(palabra, word)
    if (confianza>=0.92 and confianza>=radio):
      radio=confianza
      palabra_encontrada=word
  return palabra_encontrada

def tratamiento_texto(texto):
  trans = str.maketrans('áéíóú','aeiou')
  texto = texto.lower()
  texto = texto.translate(trans)
  texto = re.sub(r"[^\w\s]", '', texto)
  texto = " ".join(texto.split())
  return texto

#Función para normalizar la palabra
def normalizar(texto):
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

# %% [markdown]
# # 3. Cargar bases de conocimiento

# %%

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

#Importando bases de dialogo fluído

txt_folder_path = os.path.join(ruta,'txt')
lista_documentos=[x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
lista_dialogos, lista_dialogos_respuesta, lista_tipo_dialogo = [],[],[]
for idx in range(len(lista_documentos)):
  f=open(txt_folder_path+'/'+lista_documentos[idx], 'r', encoding='utf-8', errors='ignore')
  flag,posicion = True,0
  for line in f.read().split('\n'):
    if flag:
      line = tratamiento_texto(line)
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

# %% [markdown]
# # 4. Buscar respuesta del Chatbot

# %%
#Función para verificar si el usuário inició un diálogo
def dialogo(user_response):
  df = df_dialogo.copy()
  vectorizer = TfidfVectorizer()
  dialogos_numero = vectorizer.fit_transform(df_dialogo['dialogo'])
  respuesta_numero = vectorizer.transform([user_response])
  for idx,row in df.iterrows():
    df.at[idx,'interseccion'] = len(set(user_response.split()) & set(row['dialogo'].split()))/len(user_response.split())
    df.at[idx,'similarity'] = cosine_similarity(dialogos_numero[idx], respuesta_numero)[0][0]
    df.at[idx,'jaro_winkler'] = jellyfish.jaro_winkler(user_response,row['dialogo'])
    df.at[idx,'probabilidad'] = max(df.at[idx,'interseccion'],df.at[idx,'similarity'],df.at[idx,'jaro_winkler'])
  df.sort_values(by=['probabilidad','jaro_winkler'], inplace=True, ascending=False)
  probabilidad = df['probabilidad'].head(1).values[0]
  tipo = df['tipo'].head(1).values[0]
  if probabilidad >= 0.90 and tipo not in ['ElProfeAlejo']:
    #print('Respuesta encontrada por el método de comparación de textos - Probabilidad: ', probabilidad)
    respuesta = df['respuesta'].head(1).values[0]
  else:
    respuesta = ''
  return respuesta

#Cargar el modelo entrenado
ruta_modelo = os.path.join(ruta,'modelo_nuevo')

Modelo_TF = BertForSequenceClassification.from_pretrained(ruta_modelo)
tokenizer_TF = BertTokenizer.from_pretrained(ruta_modelo)

def clasificacion_modelo(pregunta):
  frase = normalizar(pregunta)
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

  diccionario = {0: 'Agradecimiento', 1: 'Aprendizaje', 2: 'Contacto', 3: 'Continuacion', 4: 'Despedida', 5: 'Edad', 6: 'ElProfeAlejo', 7: 'Error', 8: 'Funcion', 9: 'Identidad', 10: 'Nombre', 11: 'Origen', 12: 'Otros', 13: 'Saludos', 14: 'Sentimiento', 15: 'Usuario'}
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
  #print('Respuesta encontrada por el modelo que pertenece a la clase ', clase_encontrada)
  return df['respuesta'][indice_pregunta_proxima]

def respuesta_chatbot(pregunta):
  respuesta = dialogo(pregunta)
  if respuesta != '':
    return respuesta
  else:
    respuesta = clasificacion_modelo(pregunta)
    if respuesta != '':
      return respuesta
    else:
      return 'Respuesta no encontrada'

def preguntar():
  while True:
    pregunta = input(f"Preguntame: ")
    print(respuesta_chatbot(pregunta))
    if (pregunta=="Q"):
      break 

if __name__ == "__main__":
  preguntar()