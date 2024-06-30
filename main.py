#importe de librerias
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel
#from pepe import recomendacion_juego
#from pepe import developer
app = FastAPI()



@app.get("/")
def read_root():
    a="Hola Mundo"
    return {a}



#cargo el dataset
df_datos = pd.read_csv('datasets/datos.csv')
df_reviws = pd.read_csv('datasets/UserReviews.csv')
df_items = pd.read_csv('datasets/UsersItems.csv')







def recomendacion_juego( id_producto ):

    #paso todos precios a float, los que son F2P pasan a NaN
    df_datos.loc[:,'price']=pd.to_numeric(df_datos['price'], errors='coerce')
    #paso los NaN a 0
    df_datos.loc[:,'price']=df_datos['price'].fillna(0)
    #paso todo a minusculas
    df_datos.loc[:,'developer'] = df_datos['developer'].str.lower()
    #paso la columna a a formato de fecha
    df_datos['release_date'] = pd.to_datetime(df_datos['release_date'],errors='coerce')

    #transformo las columnas a srt para poder tabajarlas
    generos=df_datos['genres'].astype(str)
    tags=df_datos['tags'].astype(str)
    specs=df_datos['specs'].astype(str)
    

    vec=TfidfVectorizer()

    #creo matriz
    vec_matrix1= vec.fit_transform(generos)
    vec_matrix2= vec.fit_transform(tags)
    vec_matrix3= vec.fit_transform(specs)

    #uno las matrises
    matrix_completa=np.column_stack([vec_matrix1.toarray(),vec_matrix3.toarray(),vec_matrix2.toarray()])

    #calculo la similituid del coseno
    coseno=cosine_similarity(matrix_completa)
    #id_producto=248820.0
    #buaca el juego en el dataFrame
    juego_en_data= df_datos[df_datos['id']== id_producto]

    if not juego_en_data.empty:
        juego_indice=juego_en_data.index[0]
        #obtengo los similares
        juegos_similares= coseno[juego_indice]
        #los ordeno de mayor a menor
        juegos_mas_similares=np.argsort(-juegos_similares)
        #obtengo los 6 primeros
        top_5_juegos=df_datos.loc[juegos_mas_similares[0:6],'app_name']
        #los combierto en lista 
        top_5_juegos_mostrar=top_5_juegos.to_numpy().tolist()
        #tomo quito el primer valor para guardarlo en una variable para mostrar el nombre del juego que ingrese por id
        nombre_del_juego= top_5_juegos_mostrar.pop(0)
        a= (f'los 5 juegos recomendados para el id {id_producto} ({nombre_del_juego}) son: {top_5_juegos_mostrar}' )
        #return print(f'los 5 juegos recomendados para el id {id_producto} ({nombre_del_juego}) son: {top_5_juegos_mostrar}' )
        return a
    else:
        a='el juego no esta en la base de datos'
        return a

#creo un df con las variables a trabajar
df_auxi1= df_datos[['release_date','price','developer','id']]

def best_developer_year(ano):
    #creo el primer dataframe con las columnas a trabajar
    df_auxi11= df_datos[['release_date','id','developer']]
    #transformo la columna id a entero
    df_auxi11['id'] = df_auxi11['id'].astype(int)
    #creo la columna año
    df_auxi11['Año']=df_auxi11['release_date'].dt.year
    #borro la columna donde estan las fechas enteras
    df_auxi11=df_auxi11.drop(columns='release_date')
    #creo el segundo df
    df_auxi2 = df_reviws[['item_id','recommend','sentiment_analysis']]
    #renombro la columna item id para poder uni bien los dos dataframes
    df_auxi2.rename(columns={'item_id':'id'}, inplace=True)
    #creo un df uniendo los dataframs por la columna id
    df_auxi3 = pd.merge(df_auxi11,df_auxi2,on='id')
    #elimino los nulos
    df_auxi3.dropna(inplace=True)
    #paso la columna año a entero
    df_auxi3['Año'] = df_auxi3['Año'].astype(int)
    #quito todas las reviews negativas o neutras
    df_auxi3= df_auxi3[(df_auxi3['sentiment_analysis']!=0)&(df_auxi3['sentiment_analysis']!=1)]
    #quito todos los no recomendados
    df_auxi3= df_auxi3[(df_auxi3['recommend']!=False)]
    #elimino la columna id
    df_auxi3.drop(columns='id',inplace=True)
    #agrupo los años, los developers y las recomendaciones True, ademas sumo todas las recomendaciones positivas, cada una vale 2
    df_auxi3=df_auxi3.groupby(['developer','Año', 'recommend'])['sentiment_analysis'].sum().reset_index()
    #divido las reviews positivas para tener el numero esacto de reviews positivas
    df_auxi3['sentiment_analysis']= (df_auxi3['sentiment_analysis']/2).round().astype(int)
    #creo un dataframe flitrando por año
    df_auxi4 = df_auxi3[df_auxi3['Año']==ano]
    #lo ordeno de forma asendiente para poder obtener a los 3 mas reviews
    df_auxi4 = df_auxi4.sort_values(by='sentiment_analysis', ascending=False).reset_index(drop=True)
    #evito errores que se generan cuando no hay un minimo de 3 developers para ese año
    puesto_1 = df_auxi4['developer'].iloc[0] if len(df_auxi4) > 0 else 'Este puesto está vacío'
    puesto_2 = df_auxi4['developer'].iloc[1] if len(df_auxi4) > 1 else 'Este puesto está vacío'
    puesto_3 = df_auxi4['developer'].iloc[2] if len(df_auxi4) > 2 else 'Este puesto está vacío'
    #creo el diccionario de salida
    dicc_salida = [{'Puesto 1':puesto_1},{'Puesto 2':puesto_2},{'Puesto 3':puesto_3}]
    return dicc_salida

def developer_reviews_analysis1(desarrolador):
    #creo los dataframe con las columnas atrabajar
    df_ra=df_reviws[['item_id','sentiment_analysis']]
    df_ra2=df_datos[['id','developer']]
    #paso los id a eenteros
    df_ra2['id']=df_ra2['id'].astype(int)
    #renombro las columnas para que coinsidan
    df_ra.rename(columns={'item_id':'id'}, inplace=True)
    #uno los dos dataframes
    df_ra3=pd.merge(df_ra,df_ra2,on='id')
    #Quito la columna id
    df_ra3 = df_ra3.drop(columns='id')
    #elimino las filas con reviews neutras
    df_ra3 = df_ra3[(df_ra3['sentiment_analysis']!=1)]
    #Agrupo los desarrolladores con y creo los valores para las cantidad de reviews positivas y negativas
    df_positivos = df_ra3[df_ra3['sentiment_analysis'] == 2].groupby('developer').size().reset_index(name='Positivos')
    df_negativos = df_ra3[df_ra3['sentiment_analysis'] == 0].groupby('developer').size().reset_index(name='Negativos')
    # Combinar los dos df recien cerados
    df_final = pd.merge(df_positivos, df_negativos, on='developer', how='outer').fillna(0)
    #paso los valores a entero
    df_final['Positivos'] = df_final['Positivos'].astype(int)
    df_final['Negativos'] = df_final['Negativos'].astype(int)
    #Creo un dataframe filtrando al desarrollador que presiso y reseteo el indice
    df_salida=df_final[df_final['developer']==desarrolador].reset_index()
    #Armo una lista con los valores corespondientes para el desarrolador en cuestion
    negativos=df_salida['Negativos'].loc[0]
    positivos=df_salida["Positivos"].loc[0]
    lista_salida = ['Negativos = {}'.format(negativos),'Positivos = {}'.format(positivos)]
    return lista_salida

def developer2(desarrollador):
     # Filtro el DataFrame por el desarrollador especificado
    df_desarrollador = df_auxi1[df_auxi1['developer'] == desarrollador].copy()
    #me aseguro que la columna este en fomato fecha
    df_desarrollador['release_date'] = pd.to_datetime(df_desarrollador['release_date'], errors='coerce') 
    df_desarrollador.loc[:,'año'] = df_desarrollador['release_date'].dt.year
    
    # Agruparpo por año
    agrupado = df_desarrollador.groupby('año')
    
    # Calculo la cantidad de ítems por año
    Cantidad_Items = agrupado.size()
    
    # Calculo la cantidad de ítems free por año
    cantidad_free = agrupado.apply(lambda x: (x['price'] == 0.00).sum())
    
    # Calcular el porcentaje de ítems con precio cero por año
    cantidad_free_porsentaje = (cantidad_free / Cantidad_Items) * 100
    
    # Crear un DataFrame con los resultados
    resultado = pd.DataFrame({'Cantidad de Items': Cantidad_Items,'Contenido Free': cantidad_free_porsentaje}).reset_index()

    resultado['Contenido Free'] = resultado['Contenido Free'].apply(lambda x: f'{x:.2f}%')
    #results = results.reset_index(drop=True)
    resultado_final=resultado.to_dict(orient='index')
    return resultado_final

def user_data1(user_id):
    lista=[]
    #Filtra por usuario
    for i,j in zip(df_items['user_id'],df_items['item_id']) :
        if i == user_id:
            lista.append(j)
    #Obtiene el presio de los juegos que tenga y los suma
    precios= 0
    for i, j in zip(df_datos['id'],df_datos['price']):
        if i in lista:
            precios+=j 
    dinero_gastado = '{} USD'.format(int(precios))
    #Obtiene la contidad de recomendaciones
    rsi=0
    rno=0
    total=0
    for i, j in zip(df_reviws['user_id'],df_reviws['recommend']):
        if i == user_id:
            if j== True:
                rsi+=1
                total+=1
            else:
                rno+=1
                total+=1
    if total > 0:
        porsentaje = int((rsi/total)*100)
    else:
        porsentaje = 0
    porsentaje = '{}%'.format(porsentaje)
    #Obtiene la cantidad de items
    total_items=0
    for i, j in zip(df_items['user_id'],df_items['items_count']):
        if i == user_id:
            total_items= j
        if total_items > 0:
            break
    #retorno de la funcion en formato diccionario
    dic1 ={'usuario': user_id,
           'Dinero gastado': dinero_gastado,
           'Porsentaje de recomendacion':porsentaje,
           'Cantidad de items': total_items
           }
    return dic1

def UserForGenre(genero):
    # Asegurarse de que la fecha esté en formato datetime
    df_datos['release_date'] = pd.to_datetime(df_datos['release_date'], errors='coerce')
    
    # Filtrar los juegos que pertenecen al género especificado
    juegos_genero = df_datos[df_datos['genres'].str.contains(genero, na=False)]
    
    # Crear un DataFrame con los juegos filtrados y sus años de lanzamiento
    df_juegos = juegos_genero[['id', 'release_date']].copy()
    df_juegos['año'] = df_juegos['release_date'].dt.year
    df_juegos = df_juegos.drop(columns=['release_date']).rename(columns={'id': 'item_id'})
    
    # Filtrar las jugadas correspondientes a los juegos del género
    df_jugadas = df_items[df_items['item_id'].isin(df_juegos['item_id'])]
    
    # Unir los DataFrames de juegos y jugadas
    new_df = pd.merge(df_juegos, df_jugadas, on='item_id')
    
    # Convertir playtime_forever de minutos a horas, redondear y convertir a entero
    new_df['playtime_forever'] = (new_df['playtime_forever'] / 60).round().astype(int)
    
    # Eliminar la columna item_id
    new_df = new_df.drop(columns=['item_id'])
    
    # Agrupar por usuario y sumar las horas jugadas
    filtro_usuario = new_df.groupby('user_id')['playtime_forever'].sum().reset_index()
    
    # Ordenar el DataFrame por horas jugadas en forma descendente
    filtro_usuario = filtro_usuario.sort_values(by='playtime_forever', ascending=False).reset_index(drop=True)
    
    # Obtener el usuario con más horas jugadas
    usuario = filtro_usuario.loc[0, 'user_id']
    
    # Agrupar por año y usuario, sumando las horas jugadas
    new_df_agrupado = new_df.groupby(['año', 'user_id'])['playtime_forever'].sum().reset_index()
    
    # Filtrar solo las filas del usuario con más horas jugadas
    new_filtrado = new_df_agrupado[new_df_agrupado['user_id'] == usuario]
    
    # Ordenar el DataFrame por año en forma descendente
    new_filtrado = new_filtrado.sort_values(by='año', ascending=False).reset_index(drop=True)
    
    # Crear una lista con los años y las horas jugadas
    lista_aux = new_filtrado[['año', 'playtime_forever']].rename(columns={'playtime_forever': 'Horas'}).to_dict(orient='records')
    
    # Crear el diccionario de salida
    diccionario_salida = {
        f'Usuario con más horas jugadas para Género {genero}': usuario,
        'Horas jugadas': lista_aux
    }

    return diccionario_salida

@app.get("/user_data/{user_id}")
def user_data(user_id:str):
    return {'respuesta':user_data1(user_id)}

@app.get("/developer_reviews_analysis/{Desarrollador}")
def developer_reviews_analysis(desarrollador:str):
    
    return {desarrollador : developer_reviews_analysis1(desarrollador)}

@app.get("/best_developer_year/{Año}")
def mejor_desarrollador_año(año:int):
    return {'respuesta':best_developer_year(año)}

@app.get("/user_for_genre/{genero}")
def usuario_por_genero(genero:str):
    return {'respuesta':UserForGenre(genero)}

@app.get("/items/{item_id}")
def read_item(item_id: float):
    return {'respuesta':recomendacion_juego(item_id)}


@app.get("/developer/{desarrollador}")
def developer(desarrollador:str):
    desarrollador = desarrollador.lower()
    resultados = developer2(desarrollador)
    return{desarrollador:resultados}