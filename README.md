steps para i - run

i download yung yung 4 files

tas punta kayo sa users folder nyo then gawa kayo dalawang folder named '.streamlit' at 'resilio-map' EXACTLY AH

i move nyo yung 'credentials.toml' at 'config.toml' sa '.streamlit' na folder i move nyo yung 'app.py' at 'requirements.txt' sa 'resilio-map' na folder

tas open mo terminal/cmd

kung nasa users ka type -> cd resilio-map

tas install all libraries with -> pip install -r requirements.txt/python -m pip install requirements.txt (iinstall niya lahat ng kailangang library sa device nyo)

after mag install

type sa same command line -> streamlit run app.py or python -m streamlit run app.py tas bubukas na yan


tapos after mag run pwede mo muna paglaruan lagay mo yung dataset, after neto close muna yung app, then sa resilio-map folder, may lalabas jan na 'data' at 'venv' folder automatic gagawin ng app para sa database, sa data na folder gagawa ka ng bagong folder named 'bioclim' tas sa loob ng bioclim folder dun ilalagay yung bioclim data and may naming convention sha:

So example yung wc2.1_30s_bio_.tif -> BIO1.tif then yung wc2.1_30s_bio_4.tif -> BIO4.tif and so on hanggang sa BIO15.tif tas yun run na ulet 

tas explain lang yung purpose ng use for this session and save to database sa pag uupload ng dataset sa app, yung use for this session para sha sa mga gusto na magtest ng bagong data pag ayaw nila ihalo yung data na yun sa current database, tas duh yung save to database magsasave na yung data ng bagong csv sa database. Etong 10k dapat save to database ah.


Additional
tapos ctrl + c sa terminal para i close

btw yung mga species dito hardcoded kaya wala pang option mag insert ng csv he yung climate same din may bug yung side bar kaya wag nyo pipindutin yung button sa taas ng Resilio-Map na text pang sarado ng sidebar yun pag pinindot nyo di nyo na makikita yung button ulit mastuck kayo, ginawa kong fixed yung sidebar kala ko maakakatulong, di pala goddamit
