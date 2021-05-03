conf = None

###########################################
## MODIFICAR ESTE VALOR
bandwidth = 500
###########################################

if bandwidth == 500:
    conf = {
    'filtroSeno':'FFT',         # Filtro utilizado para eliminar ruido sinusoidal (IIR FIR FFT)
    'interFrameThreshold':290, # Umbral de MSE entre frames, mientras mayor sea, menos cuandrados se envian.
    'interFrameNeighbors': 3,
    'interFrameSendAll': 25,   # Cada cuantos frames se envia el frame completo
    'interFrameBlur':0.5,       # Cuanto afecta los cuadrados nuevos a los vecinos anteriores
    'interFrameBlurRadius':0,   # A cuantos pixeles vecinos aplica blur
    'skipFrameAfter':4,         # Cada cuantos frames salta uno (3 = 20fps, 4 = 24 fps, ...)
    'postQuantization':'Truncate',    # Método para reducir tamaño de la DCT ( RunLength o Truncate )
    'dynamicDendrogram':False,         # Si es True Genera un dendrograma por cada frame y lo envía
    'allQ':30,
    'maxQ':25,
    'minQ':18,
    'dendrograma':'dendrogramas/Dendrograma500.txt',
    'bandwidth':bandwidth
    }
elif bandwidth == 1000:
    conf = {
    'filtroSeno':'FFT',         # Filtro utilizado para eliminar ruido sinusoidal (IIR FIR FFT)
    'interFrameThreshold':145, # Umbral de MSE entre frames, mientras mayor sea, menos cuandrados se envian.
    'interFrameNeighbors':2,
    'interFrameSendAll': 20,   # Cada cuantos frames se envia el frame completo
    'interFrameBlur':0.5,       # Cuanto afecta los cuadrados nuevos a los vecinos anteriores
    'interFrameBlurRadius':0,   # A cuantos pixeles vecinos aplica blur
    'skipFrameAfter':4,         # Cada cuantos frames salta uno (3 = 20fps, 4 = 24 fps, ...)
    'postQuantization':'Truncate',    # Método para reducir tamaño de la DCT ( RunLength o Truncate )
    'dynamicDendrogram':False,         # Si es True Genera un dendograma por cada frame y lo envía
    'allQ':40,
    'maxQ':30,
    'minQ':20,
    'dendrograma':'dendrogramas/Dendrograma1000.txt',
    'bandwidth':bandwidth
    }
elif bandwidth == 5000:
    conf = {
    'filtroSeno':'FFT',         # Filtro utilizado para eliminar ruido sinusoidal (IIR FIR FFT)
    'interFrameThreshold':30, # Umbral de MSE entre frames, mientras mayor sea, menos cuandrados se envian.
    'interFrameNeighbors':3,
    'interFrameSendAll': 20,   # Cada cuantos frames se envia el frame completo
    'interFrameBlur':0.5,       # Cuanto afecta los cuadrados nuevos a los vecinos anteriores
    'interFrameBlurRadius':0,   # A cuantos pixeles vecinos aplica blur
    'skipFrameAfter':4,         # Cada cuantos frames salta uno (3 = 20fps, 4 = 24 fps, ...)
    'postQuantization':'Truncate',    # Método para reducir tamaño de la DCT ( RunLength o Truncate )
    'dynamicDendrogram':False,         # Si es True Genera un dendograma por cada frame y lo envía
    'allQ':50,
    'maxQ':45,
    'minQ':30,
    'dendrograma':'dendrogramas/Dendrograma1000.txt',
    'bandwidth':bandwidth
    }
