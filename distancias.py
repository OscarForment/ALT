import numpy as np

def levenshtein_matriz(x, y, threshold=None):
    # esta versión no utiliza threshold, se pone porque se puede
    # invocar con él, en cuyo caso se ignora
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]

def levenshtein_edicion(x, y, threshold=None):
    # a partir de la versión levenshtein_matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    #Ahora tenemos la matriz en D
    camino=[]
    i=lenX
    j=lenY
    while i>0 and j>0: #Recorremos la matriz desde el final
        min_move=min(
                D[i - 1][j],
                D[i][j - 1],
                D[i - 1][j - 1],
            )
        if D[i-1][j]==min_move and D[i][j]==D[i-1][j]+1: #comprobamos si es borrado teniendo en cuenta que siempre tiene coste 1
            camino.append((x[i-1],""))
            i-=1
        elif D[i][j-1]==min_move and D[i][j]==D[i][j-1]+1: #comprobamos si es inserción teniendo en cuenta que siempre tiene coste 1
            camino.append(("",y[j-1]))
            j-=1
        else: #sino pues es sustición
            camino.append((x[i-1],y[j-1]))
            i-=1
            j-=1
    while i>0: #por si solo quedan operaciones de borrado
        camino.append((x[i-1],""))
        i-=1
    while j>0: #por si solo quedan operaciones de inserción
        camino.append(("",y[j-1]))
        j-=1
       
    camino.reverse()
    return D[lenX, lenY],camino

def levenshtein_reduccion(x, y, threshold=None):
    # completar versión con reducción coste espacial
    lenX, lenY = len(x), len(y)
    cprev = np.zeros(lenX+1,int)
    ccurrent = np.zeros(lenX+1,int)
    for j in range(1, lenX + 1):#inicializamos el segundo vector como si fuera el primero puesto que se va a copiar
        ccurrent[j] = ccurrent[j - 1] + 1
    for i in range (1,lenY+1):#se recorre toda la palabra final
        cprev,ccurrent=ccurrent,cprev
        ccurrent[0] = cprev[0] + 1 #en la fila 0 solo se puede hacer inserción que tiene coste 1
        for j in range(1, lenX + 1):
            ccurrent[j] = min(
                cprev[j] + 1,
                ccurrent[j - 1] + 1,
                cprev[j - 1] + (x[j - 1] != y[i - 1]),
            )

    return ccurrent[lenX] # COMPLETAR Y REEMPLAZAR ESTA PARTE

def levenshtein(x, y, threshold):
    # completar versión reducción coste espacial y parada por threshold
    lenX, lenY = len(x), len(y)

    cprev = np.zeros(lenX+1,int)
    ccurrent = np.zeros(lenX+1,int)

    for j in range(1, lenX + 1):#inicializamos el segundo vector como si fuera el primero puesto que se va a copiar
        cprev[j] = cprev[j - 1] + 1
    for i in range (1,lenY+1):#se recorre toda la palabra final
        ccurrent[0] = cprev[0] + 1 #en la fila 0 solo se puede hacer inserción que tiene coste 1     
        parada = True
        if ccurrent[0] <= threshold: parada = False
        elif ccurrent[0] == threshold and lenX - j == lenY - i: parada = False
        for j in range(1, lenX + 1):
            ccurrent[j] = min(
                cprev[j] + 1,
                ccurrent[j - 1] + 1,
                cprev[j - 1] + (x[j - 1] != y[i - 1])
            )
            if ccurrent[j] < threshold: parada = False
            elif ccurrent[j] == threshold and lenX - j == lenY - i: parada = False
            
        if parada: return threshold+1
        ccurrent, cprev = cprev, ccurrent
    return cprev[lenX] # COMPLETAR Y REEMPLAZAR ESTA PARTE

def levenshtein_cota_optimista(x, y, threshold):
    lenX, lenY = len(x), len(y)
    diccionario = {}
    for key in x:
        if key in diccionario:
            diccionario[key] += 1
        else:
            diccionario[key] = 1
    for key in y:
        if key in diccionario:
            diccionario[key] -= 1
        else:
            diccionario[key] = -1
    sumPos, sumNeg = 0, 0
    for clave in diccionario:
        if diccionario[clave] < 0:
            sumNeg += diccionario[clave]
        else:
            sumPos += diccionario[clave]
    if max(sumPos, abs(sumNeg)) < threshold:
        return levenshtein(x,y, threshold)
    else:
        return threshold + 1
         # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_restricted_matriz(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + (2 - (x[i - 2] == y[j - 1] and x[i - 1]==y[j - 2]))
            )
    return D[lenX, lenY]

def damerau_restricted_edicion(x, y, threshold=None):
    # partiendo de damerau_restricted_matriz añadir recuperar
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + (2-(x[i - 2] == y[j - 2] and x[i - 1]==y[j - 2]))
            )
    # secuencia de operaciones de edición
    camino=[]
    i=lenX
    j=lenY
    while i>0 and j>0: #Recorremos la matriz desde el final
        min_move=min(
                D[i - 1][j],
                D[i][j - 1],
                D[i - 1][j - 1],
                D[i - 2][j - 2]
            )
        if D[i-1][j]==min_move and D[i][j]==D[i-1][j]+1: #comprobamos si es borrado teniendo en cuenta que siempre tiene coste 1
            camino.append((x[i-1],""))
            i-=1
        elif D[i][j-1]==min_move and D[i][j]==D[i][j-1]+1: #comprobamos si es inserción teniendo en cuenta que siempre tiene coste 1
            camino.append(("",y[j-1]))
            j-=1
        elif D[i-2][j-2]==min_move and D[i][j]==D[i-2][j-2]+1: #Comprobamos si es una transposición teniendo en cuenta que tiene coste 1
            camino.append((x[i-2]+x[i-1],y[j-1]+y[i-2]))
            i-=2
            j-=2
        else:#sino pues es sustición
            camino.append((x[i-1],y[j-1]))
            i-=1
            j-=1
            
    while i>0: #por si solo quedan operaciones de borrado
        camino.append((x[i-1],""))
        i-=1
    while j>0: #por si solo quedan operaciones de inserción
        camino.append(("",y[j-1]))
        j-=1
       
    camino.reverse()
    return D[lenX,lenY],camino

def damerau_restricted(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    lenX, lenY = len(x), len(y)

    cprev = np.zeros(lenX+1,int)
    cprev2 = np.zeros(lenX+1,int)
    ccurrent = np.zeros(lenX+1,int)

    #Para ordenarnos mejor, ahora vamos a usar j e i tal y como se hace en los ejemplos
    for i in range(1, lenX + 1): #Recorriendo la matriz verticalmente, pues la columna se mantiene constante
        cprev[i] = cprev[i - 1] + 1 #Se inicializan los elementos de la columna inicial

    for j in range (1, lenY + 1): #Recorriendo la matriz horizontalmente
        ccurrent[0] = cprev[0] + 1 #Se inicializa la primera fila, simulando el movimiento horizontal

        parada = True
        if ccurrent[0] <= threshold: parada = False
        elif ccurrent[0] == threshold and lenX - i == lenY - j: parada = False

        for i in range(1, lenX + 1): #Se recorre en vertical y horizontal
            if i > 1 and j > 1:
                ccurrent[i] = min(
                    cprev[i] + 1, #Equivalente al coste de D en [i-1] con cprev, [j]. Movimiento derecha
                    ccurrent[i - 1] + 1, #Equivalente al coste de D en [i] con ccurrent, [j-1]. Movimiento arriba
                    cprev[i - 1] + (x[i - 1] != y[j - 1]), #Si xi != yj, se sumará 1
                    cprev2[i - 2] + 1 if (x[i - 2] == y[j - 1]) and (x[i - 1] == y[j - 2]) else 10000 #Si xi-1 == yj, yj-1 == xi
                    #Equivalente a [i -2] con cprev2, [j-2]
                )
            else:
                ccurrent[i] = min(
                    cprev[i] + 1, #Equivalente al coste de D en [i-1] con cprev, [j]. Movimiento derecha
                    ccurrent[i - 1] + 1, #Equivalente al coste de D en [i] con ccurrent, [j-1]. Movimiento arriba
                    cprev[i - 1] + (x[i - 1] != y[j - 1]) #Si xi != yj, se sumará 1
                ) 
                            
            if ccurrent[i] < threshold: parada = False
            elif ccurrent[i] == threshold and lenX - i == lenY - j: parada = False
            
        if parada: return threshold + 1
        cprev, ccurrent, cprev2 = ccurrent, cprev2, cprev
    return cprev[lenX]
    

def damerau_intermediate_matriz(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + (x[i - 2] == y[j - 1] and x[i - 1]==y[j - 2]),
                D[i - 3][j - 2] + (2 - (x[i - 3] == y[j - 1]) and (x[i - 1] == y[j - 2])),
                D[i - 2][j - 3] + (2 - (x[i - 2] == y[j - 1]) and (x[i - 1] == y[j - 3]))
            )
    return D[lenX, lenY]

def damerau_intermediate_edicion(x, y, threshold=None):
    # AMPLIACION 4
    # partiendo de matrix_intermediate_damerau añadir recuperar
    # secuencia de operaciones de edición

    # COPIAR CODIGO DE DAMERAU_INTERMEDIATE_MATRIZ

    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + (x[i - 2] == y[j - 2] and x[i - 1]==y[j - 2]),
                D[i - 3][j - 2] + 2*((x[i - 3] == y[j - 1]) and (x[i - 1] == y[j - 2])),
                D[i - 2][j - 3] + 2*((x[i - 2] == y[j - 1]) and (x[i - 1] == y[j - 3])),

            )
    
    camino=[]
    i=lenX
    j=lenY
    while i>0 and j>0: # Recorremos la matriz desde el final
        min_move=min(
                D[i-1][j],
                D[i][j-1],
                D[i-1][j-1],
                D[i-2][j-2] + 1,
                D[i-2][j-3] + 2,
                D[i-3][j-2] + 2                
            )
        if D[i-1][j]==min_move and D[i][j]==D[i-1][j]+1: #comprobamos si es borrado
            camino.append((x[i-1],""))
            i-=1
        elif D[i][j-1]==min_move and D[i][j]==D[i][j-1]+1: #comprobamos si es inserción
            camino.append(("",y[j-1]))
            j-=1
        elif D[i-1][j-1] == min_move and D[i][j] == D[i-1][j-1]: #comprobamos que es sustición
            camino.append((x[i-1],y[j-1]))
            i-=1
            j-=1
        elif D[i-2][j-2] + 1 == min_move and D[i][j] == D[i-2][j-2] + 1: # comprobamos que es un intercambio
            str1 = ""
            str2 = ""
            str1 += x[i-2]
            str1 += x[i-1]
            str2 += y[i-1]
            str2 += y[i-2]
            camino.append((str1,str2))
            i -= 2
            j -= 2
        elif D[i-2][j-3] + 2 == min_move and D[i][j] == D[i-2][j-3] + 2: # comprobamos que es un intercambio
            str1 = ""
            str2 = ""
            str1 += x[i-3]
            str1 += x[i-2]
            str2 += y[i-2]
            str2 += y[i-3]
            camino.append((str1,str2))
            i -= 2
            j -= 3
        elif D[i-3][j-2] + 2 == min_move and D[i][j] == D[i-3][j-2] + 2: # comprobamos que es un intercambio
            str1 = ""
            str2 = ""
            str1 += x[i-2]
            str1 += x[i-3]
            str2 += y[i-3]
            str2 += y[i-2]
            camino.append((str1,str2))
            i -= 3
            j -= 2
        while i>0: #por si solo quedan operaciones de borrado
            camino.append((x[i-1],""))
            i-=1
        while j>0: #por si solo quedan operaciones de inserción
            camino.append(("",y[j-1]))
            j-=1
       
    camino.reverse()
    return D[lenX, lenY],camino

    # completar versión Damerau-Levenstein intermedia con matriz
   
    
def damerau_intermediate(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    lenX, lenY = len(x), len(y)
    cprev = np.zeros(lenX+1,int)
    cprev2 = np.zeros(lenX+1,int)
    cprev3 = np.zeros(lenX+1,int)
    ccurrent = np.zeros(lenX+1,int)
    #Para ordenarnos mejor, ahora vamos a usar j e i tal y como se hace en los ejemplos
    for i in range(1, lenX + 1): #Recorriendo la matriz verticalmente, pues la columna se mantiene constante
        ccurrent[i] = ccurrent[i - 1] + 1 #Se inicializan los elementos de la columna inicial
    for j in range (1, lenY+1): #Recorriendo la matriz horizontalmente
        cprev3,cprev2 = cprev2, cprev3
        cprev2,cprev = cprev, cprev2
        cprev,ccurrent = ccurrent,cprev
        ccurrent[0] = cprev[0] + 1 #Se inicializa la primera fila, simulando el movimiento horizontal
        cprev[0] = cprev2[0] + 1 #Se inicializa la primera fila, simulando el movimiento horizontal
        for i in range(1, lenX + 1): #Se recorre en vertical y horizontal
                if i > 2:
                    ccurrent[i] = min(
                    cprev[i] + 1, #Equivalente al coste de D en [i-1] con cprev, [j]. Movimiento derecha
                    ccurrent[i - 1] + 1, #Equivalente al coste de D en [i] con ccurrent, [j-1]. Movimiento arriba
                    cprev[i - 1] + 1*(x[i - 1] != y[j - 1]), #Si xi != yj, se sumará 1
                    cprev2[i - 2] + 1 if ((x[i - 2] == y[j - 1]) and (x[i - 1] == y[j - 2])) else 10000, #Si xi-1 == yj, yj-1 == xi
                    #Equivalente a [i -2] con cprev2, [j-2]
                    cprev3[i-2] + 2 if ((x[i - 3] == y[j - 1]) and (x[i - 1] == y[j - 2])) else 10000,
                    cprev2[i-3] + 2 if ((x[i - 2] == y[j - 1]) and (x[i - 1] == y[j - 3])) else 10000,
                    )
                elif i > 1:
                    ccurrent[i] = min(
                    cprev[i] + 1, #Equivalente al coste de D en [i-1] con cprev, [j]. Movimiento derecha
                    ccurrent[i - 1] + 1, #Equivalente al coste de D en [i] con ccurrent, [j-1]. Movimiento arriba
                    cprev[i - 1] + 1*(x[i - 1] != y[j - 1]), #Si xi != yj, se sumará 1
                    cprev2[i - 2] + 1 if ((x[i - 2] == y[j - 1]) and (x[i - 1] == y[j - 2])) else 1000, #Si xi-1 == yj, yj-1 == xi
                    )
                    #Equivalente a [i -2] con cprev2, [j-2]
                else:
                    ccurrent[i] = min(
                    cprev[i] + 1, #Equivalente al coste de D en [i-1] con cprev, [j]. Movimiento derecha
                    ccurrent[i - 1] + 1, #Equivalente al coste de D en [i] con ccurrent, [j-1]. Movimiento arriba
                    cprev[i - 1] + 1*(x[i - 1] != y[j - 1]), #Si xi != yj, se sumará 1
            )

        if min(ccurrent)>threshold:
            return threshold+1 
    return ccurrent[lenX]

opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'levenshtein_r': levenshtein_reduccion,
    'levenshtein':   levenshtein,
    'levenshtein_o': levenshtein_cota_optimista,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_r':     damerau_restricted,
    'damerau_im':    damerau_intermediate_matriz,
    'damerau_i':     damerau_intermediate
}

opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}

