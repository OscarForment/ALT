import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle

class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [
        ("all", True), ("title", True), ("summary", True), ("section-name", True), ('url', False),
    ]
    def_field = 'all'
    PAR_MARK = '%'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = ['urls', 'index', 'sindex', 'ptindex', 'docs', 'weight', 'articles',
                  'tokenizer', 'stemmer', 'show_all', 'use_stemming']

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {} # hash para el indice permuterm.
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.contd=0 # Contador para la clave entera(docid) del diccionario docs
        self.weight = {} # hash de terminos para el pesado, ranking de resultados.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.conta=0 # Contador para la clave entera(artid) del diccionario artículo
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()


    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_snippet(self, v:bool):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v


    def set_stemming(self, v:bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v



    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario
        
        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario
        
        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """
        
        Recorre recursivamente el directorio o fichero "root" 
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']
        for field in self.fields:
            self.index[field[0]]={}
        file_or_dir = Path(root)
        
        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in files:
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos 
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article
                
    
    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.
        
        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.multifield y self.positional se debe ampliar el indexado


        """
        self.docs[self.contd] = filename
        for i, line in enumerate(open(filename)):
            j = self.parse_article(line)
            if(not self.already_in_index(j)):
                self.articles[self.conta]=[self.contd,i,j['title'],j['url']]
                if self.multifield:
                    for field,tok in self.fields:
                        txt = j[field]
                        if tok:
                            tokens=self.tokenize(txt)
                            pos=0
                            for token in tokens:
                                if not self.positional:
                                    if token not in self.index[field]:
                                        self.index[field][token] = []
                                    if self.conta not in self.index[field][token]:
                                        self.index[field][token].append(self.conta)
                                else:
                                    if token not in self.index[field]:
                                        self.index[field][token] = {}
                                    if self.conta not in self.index[field][token]:
                                        self.index[field][token][self.conta]=[]
                                    self.index[field][token][self.conta].append(pos)
                                pos+=1
                        else:
                            token=txt
                            if token not in self.index[field]:
                                self.index[field][token] = []
                            self.index[field][token].append(self.conta)#Duda posicional url?
                else:
                    txt = j['all']
                    tokens=self.tokenize(txt)
                    self.articles[self.conta]=[self.contd,i,j['title'],j['url']]
                    pos=0
                    for token in tokens:
                        if not self.positional:
                            if token not in self.index['all']:
                                self.index['all'][token] = []
                            if self.conta not in self.index['all'][token]:
                                self.index['all'][token].append(self.conta)
                        else:
                            if token not in self.index['all']:
                                self.index['all'][token] = {}
                            if self.conta not in self.index['all'][token]:
                                self.index['all'][token][self.conta]=[]
                            self.index['all'][token][self.conta].append(pos)
                        pos+=1
                self.urls.add(j['url'])
                self.conta+=1
        self.contd+=1
        #
        # 
        # En la version basica solo se debe indexar el contenido "article"
        #
        #
        #
        #################
        ### COMPLETAR ###
        #################

        



    def set_stemming(self, v:bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v


    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()


    def make_stemming(self):
        """

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE STEMMING.

        "self.stemmer.stem(token) devuelve el stem del token"


        """
        
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################


    
    def make_permuterm(self):
        """

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE PERMUTERM


        """
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        print('========================================')
        print('Number of indexed files: {}'.format(len(self.docs)))
        print('----------------------------------------')
        print('Number of indexed articles: {}'.format(len(self.articles)))
        print('----------------------------------------')
        print('TOKENS:')
        

        if self.multifield:
            for field, _ in self.fields:
                print("Number of tokens in ",field, ": " , len(self.index[field]))

        else:
            print("Number of tokens in article",  len(self.index['all']))

        if self.permuterm:
            print("---------------------------------------------------")
            print("PERMUTERMS: ")

            if self.multifield:
                for field, _ in self.fields:
                    print("Number of permuterms in field" , len(self.ptindex[field]))
            else:
                print("Number of permuterms in article ", len(self.ptindex))

        if self.stemming:
            print("---------------------------------------------------")
            print("STEMS: ")

            if self.multifield:
                for field, _ in self.fields:
                    print("Number of stems in  field " , len(self.sindex[field]))

            else:
                print("Number of stems in article " ,  len(self.sindex))

        if self.positional:
            print("---------------------------------------------------")
            print("Positional queries are allowed")

        else:
            print("---------------------------------------------------")
            print("Positional queries are NOT allowed")

        print("===================================================")

        



        



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        if query is None or len(query) == 0:
            return []
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################


        conectores = ['AND', 'OR', 'NOT']
        query_list = query.split()

        query_list = list(map(lambda tk: tk.split(':')[::-1] if ':' in tk else [tk], query_list))

        if len(query_list) == 1 and query not in conectores:
            return self.get_posting(*query_list[0])

        terms_postings = {}
        term_pos = 0

        for term in query_list:
            if len(term) == 1:
                if term[0] not in conectores:
                    terms_postings[term_pos] = self.get_posting(*term)
            else:
                terms_postings[term_pos] = self.get_posting(*term)

            term_pos = term_pos + 1

        query_list = query.split()

        x = 0
        while x < len(query_list) - 1:

            if query_list[x] == 'NOT':
                terms_postings[x + 1] = self.reverse_posting(terms_postings.get(x + 1))

            elif query_list[x] == 'AND':
                prev_term_posting = terms_postings.get(x - 1)

                if query_list[x + 1] == 'NOT':
                    second_term_posting = self.reverse_posting(terms_postings.get(x + 2))
                    terms_postings[x + 2] = self.and_posting(prev_term_posting, second_term_posting)
                    x += 1

                else:
                    terms_postings[x + 1] = self.and_posting(prev_term_posting, terms_postings.get(x + 1))

            elif query_list[x] == 'OR':
                prev_term_posting = terms_postings.get(x - 1)

                if query_list[x + 1] == 'NOT':
                    second_term_posting = self.reverse_posting(terms_postings.get(x + 2))
                    terms_postings[x + 2] = self.or_posting(prev_term_posting, second_term_posting)
                    x += 1 

                else:
                    terms_postings[x + 1] = self.or_posting(prev_term_posting, terms_postings.get(x + 1))

            x += 1

        return terms_postings[len(query_list) - 1]
        





    def get_posting(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list
        
        NECESARIO PARA TODAS LAS VERSIONES

        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        """ index = self.index[field] if self.multifield else self.index """

        if field is not None:
            index = self.index[field]
        else:
            index = self.index['all']
        if '*' in term or '?' in term:
            return self.get_permuterm(term)

        elif self.use_stemming:
            return self.get_stemming(term, field)

        if index.get(term) is None:
            return []

        else:
            res_repetidos= list(index.get(term))
            res = []

            for i in res_repetidos:
                if i not in res:
                    res.append(i)
            return res



    def get_positionals(self, terms:str, index):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        pass

       
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################


    def get_stemming(self, term:str, field: Optional[str]=None):
        """

        Devuelve la posting list asociada al stem de un termino.
        NECESARIO PARA LA AMPLIACION DE STEMMING

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        
        stem = self.stemmer.stem(term)

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def get_permuterm(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################
        pass



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        allarts = []
        artids = list(self.articles.keys())
        artids.sort()
        for article_id in artids:
            allarts.append((self.articles[article_id][0], article_id))
        result = []
        i = 0
        j = 0
        while (i < len(p)) & (j < len(allarts)):
            if p[i] == allarts[j][1]:
                i = i + 1
                j = j + 1
            elif p[i] < allarts[j][1]:
                i = i + 1
            else:
                result.append(allarts[j][1])
                j = j + 1
        
        while j < len(allarts):
            result.append(allarts[j][1])
            j = j + 1

        return result
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################



    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        result = []
        i = 0
        j = 0
        while (i < len(p1)) & (j < len(p2)):
            if p1[i] == p2[j]:
                result.append(p2[j])
                i = i + 1
                j = j + 1
            elif p1[i] < p2[j]:
                i = i + 1
            else:
                j = j + 1

        return result
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################



    def or_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 o p2

        """

        result = []
        i = 0
        j = 0
        while (i < len(p1)) & (j < len(p2)):
            if p1[i] == p2[j]:
                result.append(p2[j])
                i = i + 1
                j = j + 1
            elif p1[i] < p2[j]:
                result.append(p1[i])
                i = i + 1
            else:
                result.append(p2[j])
                j = j + 1
        
        while i < len(p1):
            result.append(p1[i])
            i = i + 1

        while j < len(p2):
            result.append(p2[j])
            j = j + 1

        return result
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################


    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """
        result = []
        ind1 = 0
        ind2 = 0

        while ind1 < len(p1) and ind2 < len(p2):
            if p1[ind1] == p2[ind2]:
                ind1 += 1
                ind2 += 1
            elif p1[ind1] < p2[ind2]:
                result.append(p1[ind1])
                ind1 += 1
            else:
                p2_index += 1

        while ind1 < len(p1):
            result.append(p1[ind1])
            ind1 += 1

        return result





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True                    
            else:
                """print(query)"""
        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        ################
        ## COMPLETAR  ##
        ################

        resultado = self.solve_query(query)

        if self.use_ranking:
            resultado = self.rank_result(resultado, query)

        print("===================================================")
        print("Query: ", query)
        if not self.show_all:
            result=[]
            print("Found in articles (only the first 10 articles): ")
            i=0
            while i<10 and i<len(resultado): 
                print("    #", i+1, "(", resultado[i], ")", self.articles[resultado[i]][2],":         ", self.articles[resultado[i]][3])
                i+=1
            
        else:
            print("Found in articles: ")
            for i in range(len(resultado)):
                print("    #", i+1, "(", resultado[i], ")", self.articles[resultado[i]][2],":         ", self.articles[resultado[i]][3])
        
        print("=================================================")
        print("Number of results:", len(resultado))







        

