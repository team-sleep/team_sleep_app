# import hydralit automatically imports all of Streamlit
# https://github.com/TangleSpace/hydralit
import hydralit as st
#import psycopg2  #for postgres
import pandas as pd

app = st.HydraApp(title='Team Sleep App', nav_horizontal=True, 
      use_banner_images=[None,None,{'header':"<h1 style='text-align:center;padding: 0px 0px;color:black;font-size:200%;'>App title here</h1><br>"},None,None], 
      banner_spacing=[5,30,60,30,5],)

@app.addapp(title="User", is_home=True) #set to false to not use "home icon" for this page
def tab1():
	from pages import tab1
	tab1.run()

@app.addapp(title="Model")
def tab2():
	from pages import tab2
	tab2.run()

@app.addapp(title="Temporary")
def tab_temporary_put_on_tab2_later():
	from pages import tab3
	tab3.run()

#Hydralit: navbar, state management and app isolation, all with this tiny amount of work.
app.run()