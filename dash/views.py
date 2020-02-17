from django.shortcuts import render, redirect
from django.template.loader import get_template
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.decorators import login_required
from django.core.mail import EmailMessage
from .forms import UserUpdateForm, ProfileUpdateForm, ContactForm
from rest_framework.views import APIView
from rest_framework.response import Response
import folium
from pivottablejs import pivot_ui
from .models import *
from django.db import connection
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from geopy import distance
from collections import namedtuple
from geopy.geocoders import Nominatim
import nltk


@login_required
def home(request):
    # Connections
    query_clg = str(College.objects.all().query)
    query_int = str(Internship.objects.all().query)
    df_clg = pd.read_sql_query(query_clg, connection)
    df_int = pd.read_sql_query(query_int, connection)

    # College Table--> Enrollment
    current_enrolment = df_clg['Course_year'].loc[(df_clg['Course_year'] == '2017/2018')].count()
    total_enrolment = df_clg['Student_id'].count()

    # Internship  Table--> Internships
    current_internships = df_int['Year'].loc[(df_int['Year'] == '2017/2018')].count()
    total_internships = df_int['Year'].count()

    # Internship  Table--> Hiring Entreprise
    current_enterprises = df_int['Company_Name'].loc[(df_int['Year'] == '2017/2018')].nunique()
    total_enterprises = df_int['Company_Name'].nunique()

    # Internship Table--> Average Salary
    average_salary = round(df_int['Pay_Details'].loc[(df_int['Year'] == '2017/2018')].mean(), 2)
    total_average_salary = round(df_int['Pay_Details'].mean(), 2)

    context = {
        'title': 'Home',
        'current_enrolment': current_enrolment,
        'total_enrolment': total_enrolment,
        'current_internships': current_internships,
        'total_internships': total_internships,
        'current_enterprises': current_enterprises,
        'total_enterprises': total_enterprises,
        'average_salary': average_salary,
        'total_average_salary': total_average_salary,
    }
    return render(request, 'dash/home.html', context)


@login_required
def charts(request):
    return render(request, 'dash/graph.html', {'title': 'Charts'})


@login_required
def tables_std(request):
    context = {
        'posts': Student.objects.all(),
        'title': 'Tables_student'
    }
    return render(request, 'dash/table.html',context)


@login_required
def tables_clg(request):
    context = {
        'posts': College.objects.all(),
        'title': 'Tables_college'
    }
    return render(request, 'dash/tables_clg.html',context)


@login_required
def tables_intern(request):
    context = {
        'posts': Internship.objects.all(),
        'title': 'Tables_Inernship'
    }
    return render(request, 'dash/tables_intern.html',context)


# ***************************** FOR MAPS *************************************

@login_required
def maps(request):
    return render(request, 'dash/maps.html', {'title': 'Maps'})


@login_required
def map_store(request):
    return render(request, 'map/map_store_home.html', {'title': 'Map-store'})


@login_required
def map_store_route(request):
    return render(request, 'map/map_store_route.html', {'title': 'Map-store'})


@login_required
def map_store_route1(request):
    return render(request, 'map/map_store_route1.html', {'title': 'Map-store1'})


@login_required
def map_store_route2(request):
    return render(request, 'map/map_store_route2.html', {'title': 'Map-store2'})


@login_required
def map_store_kmeans_student_city(request):
    return render(request, 'map/map_store_kmeans_student_city.html', {'title': 'Map-store'})


@login_required
def map_store_companies(request):
    return render(request, 'map/map_store_companies.html', {'title': 'Map-store'})


# ***************************** END FOR MAPS *************************************
@login_required
def map_store(request):
    return render(request, 'map/map_store_home.html', {'title': 'Map-store'})


@login_required
def graph(request):

    return render(request, 'chart-script/dash-filter.html', {'title': 'graph'})


@login_required
def graph_cergy(request):

    return render(request, 'chart-script/dash-filter-cergy.html', {'title': 'graph'})


@login_required
def piv_std(request):
    return render(request, 'piv/piv_std.html', {'title': 'Piv_std'})


@login_required
def piv_clg(request):
    return render(request, 'piv/piv_clg.html', {'title': 'Piv_clg'})


@login_required
def piv_intern(request):
    return render(request, 'piv/piv_intern.html', {'title': 'Piv_intern'})


@login_required
def visual_std(request):
    return render(request, 'dash/visu_std.html', {'title': 'Visual_std'})


@login_required
def visual_clg(request):
    return render(request, 'dash/visu_clg.html', {'title': 'Visual_clg'})


@login_required
def visual_intern(request):
    return render(request, 'dash/visu_intern.html', {'title': 'Visual_intern'})


@login_required
def regression(request):
    return render(request, 'dash/regression.html', {'title': 'Regression'})

# *****************************PAU Cergy REGRESSION MODEL FOR AVERAGE RENUMERATION *************************************
@login_required
def reg_pau(request):
    return render(request, 'chart-script/dash-filter-reg-pau.html', {'title': 'reg-pau'})


@login_required
def reg_cergy(request):
    return render(request, 'chart-script/dash-filter-reg-cergy.html', {'title': 'reg-cergy'})

# *****************************PAU Cergy REGRESSION MODEL FOR AVERAGE Enrollment *************************************
@login_required
def reg_pau_enrollment(request):
    return render(request, 'chart-script/dash-filter-reg-enrolment-pau.html', {'title': 'reg-pau'})


@login_required
def reg_cergy_enrollment(request):
    return render(request, 'chart-script/dash-filter-reg-enrolment-cergy.html', {'title': 'reg-cergy'})

# *****************************TOP 10 RECRUITERS *************************************
@login_required
def recruiters(request):
    return render(request, 'chart-script/dash-filter-top-recruiters.html', {'title': 'reg-recruiters'})


@login_required
def contact(request):
    form_class = ContactForm
    if request.method == 'POST':
        form = form_class(data=request.POST)
        if form.is_valid():
            Name = request.POST.get('Name', '')
            Email = request.POST.get('Email', '')
            Message = request.POST.get('Message', '')

            template = get_template('dash/contact_template.txt')
            context = {
                'Name': Name,
                'Email': Email,
                'Message': Message,
            }
            content = template.render(context)
            email= EmailMessage(
                "New Contact Form Submission", content,
                "YOUR Website" +'',['muralikrishnamopidevi1@gmail.com'],
                headers = {'Reply-To': Email}
            )
            email.send()
            messages.success( request , f'Message Send Successfully..!' )
            return redirect('dash-contact')
    return render(request, 'dash/contact.html', {'form': form_class}, {'title': 'Contact'})


@login_required
def profile(request):
    if request.method == 'POST':
        uu_form = UserUpdateForm(request.POST, instance=request.user)
        pp_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        if uu_form.is_valid() and pp_form.is_valid():
            uu_form.save()
            pp_form.save()
            messages.success(request, f'Your account has been updated')
            return redirect('dash-profile')
    else:
        uu_form = UserUpdateForm(instance=request.user)
        pp_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'uuu_form': uu_form,
        'pup_form': pp_form
    }
    return render(request, 'dash/profile.html', context)


@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important!
            messages.success(request, f'Your password was successfully updated!')
            return redirect('dash-profile')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'dash/password.html', {
        'form': form
    })


class ChartData(APIView):
    authentication_classes = ()
    permission_classes = ()

    # *****************************Home Page MAP*********************************************

    def get(self, request, format=None):

        def extract_entity_names(t):
            entity_names = []

            if hasattr(t, 'label') and t.label:
                if t.label() == 'NE':
                    entity_names.append(' '.join([child[0] for child in t]))
                else:
                    for child in t:
                        entity_names.extend(extract_entity_names(child))

            return entity_names

        def get_bearing(p1, p2):

            long_diff = np.radians(p2.lon - p1.lon)

            lat1 = np.radians(p1.lat)
            lat2 = np.radians(p2.lat)

            x = np.sin(long_diff) * np.cos(lat2)
            y = (np.cos(lat1) * np.sin(lat2)
                 - (np.sin(lat1) * np.cos(lat2)
                    * np.cos(long_diff)))
            bearing = np.degrees(np.arctan2(x, y))

            # adjusting for compass bearing
            if bearing < 0:
                return bearing + 360
            return bearing

        def get_arrows(locations, color='blue', size=6, n_arrows=3):

            Point = namedtuple('Point', field_names=['lat', 'lon'])

            # creating point from our Point named tuple
            p1 = Point(locations[0][0], locations[0][1])
            p2 = Point(locations[1][0], locations[1][1])

            # getting the rotation needed for our marker.
            # Subtracting 90 to account for the marker's orientation
            # of due East(get_bearing returns North)
            rotation = get_bearing(p1, p2) - 90

            # get an evenly space list of lats and lons for our arrows
            # note that I'm discarding the first and last for aesthetics
            # as I'm using markers to denote the start and end
            arrow_lats = np.linspace(p1.lat, p2.lat, n_arrows + 2)[1:n_arrows + 1]
            arrow_lons = np.linspace(p1.lon, p2.lon, n_arrows + 2)[1:n_arrows + 1]

            arrows = []

            # creating each "arrow" and appending them to our arrows list
            for points in zip(arrow_lats, arrow_lons):
                arrows.append(folium.RegularPolygonMarker(location=points,
                                                          fill_color=color, number_of_sides=3,
                                                          radius=size, rotation=rotation).add_to(some_map))
            return arrows

        # *****************************LOADING DATA FROM MODELS*************************************

        # *****************************LOADING DATA FROM MODELS*************************************

        student_table = str(Student.objects.all().query)
        student_df = pd.read_sql_query(student_table, connection)

        college_table = str(College.objects.all().query)
        college_df = pd.read_sql_query(college_table, connection)

        internship_table = str(Internship.objects.all().query)
        internship_df = pd.read_sql_query(internship_table, connection)

        country_table = str(Country.objects.all().query)
        country_df = pd.read_sql_query(country_table, connection)

        # *****************************END DATA FROM MODELS END *************************************

        # ***************************** FOLLIUM MAP *************************************

        df_clean = pd.DataFrame(student_df['Country'].unique(), columns=['country'])

        joined = pd.merge(df_clean, country_df, left_on=['country'], right_on=['country'], how='inner')

        m = folium.Map(location=[47.083333, 2.4], tiles='OpenStreetMap', zoom_control=True, zoom_start=2, min_zoom=2,)

        for i in range(0, len(joined)):

            folium.Marker([joined.iloc[i]['latitude'], joined.iloc[i]['longitude']], popup=joined.iloc[i]['country_name']).add_to(m)

        m.save('dash/templates/map/map_store_home.html')

        # ***************************** END FOLLIUM MAP END*************************************

        # *****************************END TOP 10 Recruiters Every Year END *************************************

        # ***************************** FOLLIUM HOME PAGE MAP *************************************

        df_clean = pd.DataFrame(student_df['Country'].unique(), columns=['country'])

        joined = pd.merge(df_clean, country_df, left_on=['country'], right_on=['country'], how='inner')

        m = folium.Map(location=[47.083333, 2.4], tiles='OpenStreetMap', zoom_control=False, zoom_start=2, no_wrap=True,
                       min_zoom=2, max_bounds=True, )

        for i in range(0, len(joined)):
            folium.Marker([joined.iloc[i]['latitude'], joined.iloc[i]['longitude']],
                          popup=joined.iloc[i]['country_name']).add_to(m)

        m.save('dash/templates/map/map_store_home.html')

        # ***************************** END FOLLIUM MAP END*************************************

        # ***************************** FOLLIUM MAP ON MAPS *************************************

        center_lat = 49.0331671
        center_lon = 2.0547222
        some_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        p1 = [48.8640493, 2.3310526]
        p2 = [49.0331671, 2.0547222]

        dist = distance.distance(p1, p2).km

        folium.Marker(location=p1, icon=folium.Icon(color='green')).add_to(some_map)

        folium.Marker(location=p2, icon=folium.Icon(color='red')).add_to(some_map)

        folium.PolyLine(locations=[p1, p2], color='blue').add_to(some_map)

        arrows = get_arrows(locations=[p1, p2], n_arrows=3)

        for arrow in arrows:
            arrow.add_to(some_map)

        some_map.save('dash/templates/map/map_store_route.html')

        # ***************************** END FOLLIUM MAP END*************************************
        # ***************************** FOLLIUM MAP ON MAPS *************************************

        center_lat = 43.6051074
        center_lon = 1.3903425
        some_map1 = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        p1 = [43.6051074, 1.3903425]
        p2 = [43.31726632, -0.345934294]

        dist1 = distance.distance(p1, p2).km

        folium.Marker(location=p1, icon=folium.Icon(color='green')).add_to(some_map1)

        folium.Marker(location=p2, icon=folium.Icon(color='red')).add_to(some_map1)

        folium.PolyLine(locations=[p1, p2], color='blue').add_to(some_map1)

        arrows = get_arrows(locations=[p1, p2], n_arrows=3)

        for arrow in arrows:
            arrow.add_to(some_map1)

        some_map1.save('dash/templates/map/map_store_route1.html')

        # ***************************** END FOLLIUM MAP END*************************************
        # ***************************** FOLLIUM MAP ON MAPS *************************************

        center_lat = 49.0331671
        center_lon = 2.0547222
        some_map2 = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        p1 = [49.0331671, 2.0547222]
        p2 = [48.93831742, 2.180910161]

        dist2 = distance.distance(p1, p2).km

        folium.Marker(location=p1, icon=folium.Icon(color='green')).add_to(some_map2)

        folium.Marker(location=p2, icon=folium.Icon(color='red')).add_to(some_map2)

        folium.PolyLine(locations=[p1, p2], color='blue').add_to(some_map2)

        arrows = get_arrows(locations=[p1, p2], n_arrows=3)

        for arrow in arrows:
            arrow.add_to(some_map2)

        some_map2.save('dash/templates/map/map_store_route2.html')

        # ***************************** END FOLLIUM MAP END*************************************
        # ***************************** FOLLIUM MAP ENTERPRISE *************************************

        datafr = pd.read_csv('Enterprise_Co_ord.csv')

        datafr['fol_label'] = datafr['Company_Name'].str.cat(datafr['City'], sep=' ::  ')
        datafr[['Latitude', 'Longitude', 'fol_label']].head()

        datafr[['Latitude', 'Longitude', 'fol_label']]
        # Make an empty map
        m = folium.Map(location=[47.083333, 2.4], zoom_control=False, zoom_start=6, no_wrap=True, min_zoom=2,
                       max_bounds=True)
        for i in range(0, len(datafr)):
            folium.Marker([datafr.iloc[i]['Latitude'], datafr.iloc[i]['Longitude']], popup=datafr.iloc[i]['fol_label'],
                          icon=folium.Icon(color='green', icon='cloud')).add_to(m)

        m.save('dash/templates/map/map_store_companies.html')

        # ***************************** END FOLLIUM MAP ENTERPRISE END *************************************
        # ***************************** FOLLIUM MAP STUDENT KMEANS*************************************

        std = student_df
        std.reset_index(drop=True)
        std['City'] = std['City'].str.upper()
        std['City'] = std['City'].str.replace("0", "OTHER", case=False)
        std['City'] = std['City'].str.replace("CHOIX", "PONTOISE", case=False)
        std['City'] = std['City'].str.replace("CERGY CEDEX", "CERGY", case=False)

        std = std[std.City != 'OTHER']
        top_twenty_cities = std['City'].value_counts()[:20]

        top_twenty_cities = top_twenty_cities.reset_index()
        top_twenty_cities.columns = ['City_name', 'Count']

        std_merge = pd.merge(top_twenty_cities, std, left_on=['City_name'], right_on=['City'], how='inner')[
            ['id', 'City_name', 'Country', 'Postal_Code']]

        std_merge = std_merge.drop_duplicates(subset=['City_name'])
        std_merge.reset_index(drop=True)
        std_merge['Address'] = std_merge['City_name'] + ", " + std_merge['Postal_Code'].apply(str) + ", " + std_merge[
            'Country']
        std_merge.reset_index(drop=True, inplace=True)

        nm = Nominatim(user_agent="EISTI")
        std_merge['Coordinates'] = std_merge['Address'].apply(nm.geocode)

        std_merge["Latitude"] = std_merge["Coordinates"].apply(lambda x: x.latitude if x != None else None)
        std_merge["Longitude"] = std_merge["Coordinates"].apply(lambda x: x.longitude if x != None else None)
        city = std_merge

        cities = city[['Latitude', 'Longitude', 'City_name']]
        cities.drop(cities.iloc[:, 3:9], inplace=True, axis=1, )

        cities.info()

        cities_num = cities.values

        campuses = np.array([[49.052502, 2.038830], [43.297539, -0.374640]])

        # K means analysis
        kmeans1 = KMeans(n_clusters=2, random_state=0, init=campuses, max_iter=1).fit(cities[['Latitude', 'Longitude']])

        analysis = kmeans1.fit_predict(cities[['Latitude', 'Longitude']])

        c1 = pd.DataFrame(cities[analysis == 0])
        c2 = pd.DataFrame(cities[analysis == 1])
        c1.reset_index(drop=True)
        c2.reset_index(drop=True)

        c1['class'] = "CERGY"
        c2['class'] = "PAU"
        c3 = c1.append(c2)
        c3.reset_index(drop=True).head()

        m = folium.Map(location=[47.083333, 2.4], zoom_control=False, zoom_start=6, no_wrap=True, min_zoom=2,
                       max_bounds=True)

        # I can add marker one by one on the map
        lat = 49.052502;
        lon = 2.038830
        folium.Marker([lat, lon], icon=folium.Icon(color='green', icon='cloud')).add_to(m)

        lat = 43.297539;
        lon = -0.37464
        folium.Marker([lat, lon], icon=folium.Icon(color='black', icon='cloud')).add_to(m)

        for i in range(0, len(c1)):
            folium.Marker([c1.iloc[i]['Latitude'], c1.iloc[i]['Longitude']], popup=c1.iloc[i]['City_name'],
                          icon=folium.Icon(color='blue', icon='cloud')).add_to(m)

        for i in range(0, len(c2)):
            folium.Marker([c2.iloc[i]['Latitude'], c2.iloc[i]['Longitude']], popup=c2.iloc[i]['City_name'],
                          icon=folium.Icon(color='red', icon='cloud')).add_to(m)

        m.save('dash/templates/map/map_store_kmeans_student_city.html')

        # *****************************END FOLLIUM MAP STUDENT KMEANS END*************************************


        # ***************************** ALL GRAPHS OR CHARTS DF GOES HERE *************************************

        # Home Page - "PIE CHART"

        student_df['City'] = student_df['City'].str.upper()
        df2 = student_df.groupby(by='City', as_index=False).agg({'id': pd.Series.nunique})
        df2 = df2.sort_values(by='id', ascending=False)

        labels_pie = df2['City'][:21].tolist()
        count_of_students_pie = df2['id'][:21].tolist()

        # Home Page - "BAR CHART"

        college_df['Stream'] = college_df['Stream'].str.upper()
        df_clg2 = college_df['Stream'].value_counts().rename_axis('Stream').reset_index(name='counts')

        labels_bar = df_clg2['Stream'][:10].tolist()
        count_of_students_bar = df_clg2['counts'][:10].tolist()

        # ***************************** END ALL GRAPHS OR CHARTS DF GOES HERE END ********************

        # ***************************** Pivot_tabel GOES HERE*****************************************

        # PIVOT TABLE for Student

        pivot_ui(student_df[['Postal_Code', 'City', 'Country']], outfile_path="dash/templates/piv/piv_std.html",
                 url="dash/templates/piv/piv_std.html")

        # PIVOT TABLE for College

        pivot_ui(college_df[['Stream', 'Course_year', 'Campus']], outfile_path="dash/templates/piv/piv_clg.html",
                 url="dash/templates/piv/piv_clg.html")

        # PIVOT TABLE for Internship

        pivot_ui(internship_df[['Course', 'Year', 'Company_Name', 'City', 'Country', 'Pay_Details']],
                 outfile_path="dash/templates/piv/piv_intern.html", url="dash/templates/piv/piv_intern.html")

        # *****************************END Pivot_tabel GOES HERE END***********************************#

        # ******************************ALL CHARTS TAB CODE GOES HERE ****************************************

        # FOR BAR and PIE CHARTS

        tpau = college_df.loc[(college_df['Campus'] == 'PAU')]
        tcergy = college_df.loc[(college_df['Campus'] == 'CERGY')]
        tpauplot = internship_df[internship_df.Student_id.isin(tpau.Student_id)]
        tpauplot = tpauplot.sort_values(by='Year')
        tpauplot = tpauplot[pd.notnull(tpauplot['Year'])]
        tcergyplot = internship_df[internship_df.Student_id.isin(tcergy.Student_id)]
        tcergyplot = tcergyplot.sort_values(by='Year')
        tcergyplot = tcergyplot[pd.notnull(tcergyplot['Year'])]

        # For PAU

        dft1 = tpauplot['Year'].value_counts().rename_axis('Year').sort_index().reset_index(name='Student Intern')
        p_label = dft1['Year'].tolist()
        p_data = dft1['Student Intern'].tolist()

        # For CERGY

        dft2 = tcergyplot['Year'].value_counts().rename_axis('Year').sort_index().reset_index(name='Student Intern')
        c_label = dft2['Year'].tolist()
        c_data = dft2['Student Intern'].tolist()

        # ******************************END ALL CHARTS TAB CODE GOES HERE END**********************************

        # ***************************** FILTER GRAPH CERGY *************************************

        tpau = college_df.loc[(college_df['Campus'] == 'CERGY')]

        tpauplot = internship_df[internship_df.Student_id.isin(tpau.Student_id)]
        tpauplot = tpauplot.sort_values(by='Year')
        tpauplot = tpauplot[pd.notnull(tpauplot['Year'])]

        y_1314 = tpauplot.loc[tpauplot['Year'] == "2013/2014"]
        y_1314 = y_1314['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        y_1415 = tpauplot.loc[tpauplot['Year'] == "2014/2015"]
        y_1415 = y_1415['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        y_1516 = tpauplot.loc[tpauplot['Year'] == "2015/2016"]
        y_1516 = y_1516['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        y_1617 = tpauplot.loc[tpauplot['Year'] == "2016/2017"]
        y_1617 = y_1617['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        y_1718 = tpauplot.loc[tpauplot['Year'] == "2017/2018"]
        y_1718 = y_1718['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        # ***************************** LABELS *************************************

        y_1314_program_labels = y_1314['Program'].tolist()
        y_1415_program_labels = y_1415['Program'].tolist()
        y_1516_program_labels = y_1516['Program'].tolist()
        y_1617_program_labels = y_1617['Program'].tolist()
        y_1718_program_labels = y_1718['Program'].tolist()

        # ***************************** COUNTS *************************************

        y_1314_student_counts = y_1314['student_count'].tolist()
        y_1415_student_counts = y_1415['student_count'].tolist()
        y_1516_student_counts = y_1516['student_count'].tolist()
        y_1617_student_counts = y_1617['student_count'].tolist()
        y_1718_student_counts = y_1718['student_count'].tolist()

        # *****************************END FILTER GRAPH PAU END *****************************************

        # *****************************START FILTER GRAPH PAU START *************************************

        tpau = college_df.loc[(college_df['Campus'] == 'PAU')]

        tpauplot = internship_df[internship_df.Student_id.isin(tpau.Student_id)]
        tpauplot = tpauplot.sort_values(by='Year')
        tpauplot = tpauplot[pd.notnull(tpauplot['Year'])]

        y_1314 = tpauplot.loc[tpauplot['Year'] == "2013/2014"]
        y_1314 = y_1314['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        y_1415 = tpauplot.loc[tpauplot['Year'] == "2014/2015"]
        y_1415 = y_1415['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        y_1516 = tpauplot.loc[tpauplot['Year'] == "2015/2016"]
        y_1516 = y_1516['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        y_1617 = tpauplot.loc[tpauplot['Year'] == "2016/2017"]
        y_1617 = y_1617['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        y_1718 = tpauplot.loc[tpauplot['Year'] == "2017/2018"]
        y_1718 = y_1718['Course'].value_counts().rename_axis('Program').sort_index().reset_index(name='student_count')

        # ***************************** LABELS *************************************

        y_1314_program_labels_pau = y_1314['Program'].tolist()
        y_1415_program_labels_pau = y_1415['Program'].tolist()
        y_1516_program_labels_pau = y_1516['Program'].tolist()
        y_1617_program_labels_pau = y_1617['Program'].tolist()
        y_1718_program_labels_pau = y_1718['Program'].tolist()

        # ***************************** COUNTS *************************************

        y_1314_student_counts_pau = y_1314['student_count'].tolist()
        y_1415_student_counts_pau = y_1415['student_count'].tolist()
        y_1516_student_counts_pau = y_1516['student_count'].tolist()
        y_1617_student_counts_pau = y_1617['student_count'].tolist()
        y_1718_student_counts_pau = y_1718['student_count'].tolist()

        # *****************************END FILTER GRAPH PAU END *************************************

        # *****************************PAU REGRESSION MODEL FOR AVERAGE RENUMERATION *************************************

        tpau = college_df.loc[(college_df['Campus'] == 'PAU')]

        tpauplot = internship_df[internship_df.Student_id.isin(tpau.Student_id)]
        tpauplot = tpauplot.sort_values(by='Year')
        tpauplot = tpauplot[pd.notnull(tpauplot['Year'])]

        group = tpauplot.groupby('Year')
        aggregate_pau = group.aggregate({'Pay_Details': np.mean})

        df_aggregate_pau = pd.DataFrame(aggregate_pau)
        df_aggregate_pau = df_aggregate_pau.reset_index()
        new = df_aggregate_pau['Year'].str.split("/", n=1, expand=True)
        Xp = new[0].astype(int)
        X = np.array(Xp).reshape((-1, 1))
        Y = np.array(df_aggregate_pau['Pay_Details'])

        model = LinearRegression()
        model.fit(X, Y)
        r_sq = model.score(X, Y)
        y_hat_p1_pau = model.predict(X)

        X2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
        model.fit(X2, Y)
        r_sq = model.score(X2, Y)
        y_hat_p2_pau = model.predict(X2)

        X3 = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
        model.fit(X3, Y)
        r_sq = model.score(X3, Y)
        y_hat_p3_pau = model.predict(X3)


        xaxis_pau = df_aggregate_pau['Year'].tolist()
        yaxis_pau = Y

        # *****************************END PAU REGRESSION MODEL FOR AVERAGE RENUMERATION END*************************************

        # *****************************CERGY REGRESSION MODEL FOR AVERAGE RENUMERATION *************************************

        tcergy = college_df.loc[(college_df['Campus'] == 'CERGY')]

        tcergyplot = internship_df[internship_df.Student_id.isin(tcergy.Student_id)]
        tcergyplot = tcergyplot.sort_values(by='Year')
        tcergyplot = tcergyplot[pd.notnull(tcergyplot['Year'])]

        # MEAN PAY FOR CERGY

        aggregate_cergy = tcergyplot.groupby('Year').aggregate({'Pay_Details': np.mean})
        df_aggregate_cergy = pd.DataFrame(aggregate_cergy)
        df_aggregate_cergy = df_aggregate_cergy.reset_index()

        newc = df_aggregate_cergy['Year'].str.split("/", n=1, expand=True)
        Xc = newc[0].astype(int)
        X = np.array(Xc).reshape((-1, 1))
        Y = np.array(df_aggregate_cergy['Pay_Details'])

        model = LinearRegression()
        model.fit(X, Y)
        r_sq1_cergy = model.score(X, Y)
        y_hat_pl_cergy = model.predict(X)

        X2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
        model.fit(X2, Y)
        r_sq2_cergy = model.score(X2, Y)
        y_hat_p2_cergy = model.predict(X2)

        X3 = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
        model.fit(X3, Y)
        r_sq3_cergy = model.score(X3, Y)
        y_hat_p3_cergy = model.predict(X3)

        xaxis_cergy = df_aggregate_cergy['Year'].tolist()
        yaxis_cergy = Y

        # *****************************END CERGY REGRESSION MODEL FOR AVERAGE RENUMERATION END*************************************

        # ***************************** PAU REGRESSION MODEL FOR AVERAGE ENROLLMENT *************************************

        pau_enroll = college_df[college_df.Campus == 'PAU'].sort_values(by='Course_year')
        # t_enroll = t_enroll.replace(to_replace="2009/2009",value="2009/2010")
        penco = pau_enroll['Course_year'].value_counts()
        df_student_pau = pd.DataFrame(penco).reset_index()
        df_student_pau.columns = ['Course_year', 'Enrollments']
        df_student_pau = df_student_pau.sort_values(by='Course_year')
        df_student_pau = df_student_pau.reset_index(drop=True)

        new = df_student_pau['Course_year'].str.split("/", n=1, expand=True)
        Xp = new[0].astype(int)
        X = np.array(Xp).reshape((-1, 1))
        Y = np.array(df_student_pau['Enrollments'])

        model = LinearRegression()
        model.fit(X, Y)
        r_sq_enrollment_pau1 = model.score(X, Y)
        y_hat_p1_enrollment_pau = model.predict(X)

        X2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
        model.fit(X2, Y)
        r_sq_enrollment_pau2 = model.score(X2, Y)
        y_hat_p2_enrollment_pau = model.predict(X2)

        X3 = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
        model = LinearRegression()
        model.fit(X3, Y)
        r_sq_enrollment_pau3 = model.score(X3, Y)
        y_hat_p3_enrollment_pau = model.predict(X3)

        X4 = PolynomialFeatures(degree=4, include_bias=False).fit_transform(X)
        model = LinearRegression()
        model.fit(X4, Y)
        r_sq_enrollment_pau4 = model.score(X4, Y)
        y_hat_p4_enrollment_pau = model.predict(X4)

        xaxis_enrollment_pau = df_student_pau['Course_year'].tolist()
        yaxis_enrollment_pau = Y


        # *****************************END PAU REGRESSION MODEL FOR AVERAGE ENROLLMENT END*************************************

        # ***************************** CERGY REGRESSION MODEL FOR AVERAGE ENROLLMENT *************************************
        cergy_enroll = college_df[college_df.Campus == 'CERGY'].sort_values(by='Course_year')
        cergy_enroll = cergy_enroll.replace(to_replace="2009/2009", value="2009/2010")
        cenco = cergy_enroll['Course_year'].value_counts()
        df_student_cergy = pd.DataFrame(cenco).reset_index()
        df_student_cergy.columns = ['Course_year', 'Enrollments']
        df_student_cergy = df_student_cergy.sort_values(by='Course_year')
        df_student_cergy = df_student_cergy.reset_index(drop=True)

        new = df_student_cergy['Course_year'].str.split("/", n=1, expand=True)
        Xc = new[0].astype(int)
        X = np.array(Xc).reshape((-1, 1))
        Y = np.array(df_student_cergy['Enrollments'])

        model = LinearRegression()
        model.fit(X, Y)
        r_sq_enrollment_cergy1 = model.score(X, Y)
        y_hat_p1_enrollment_cergy = model.predict(X)

        X2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
        model = LinearRegression()
        model.fit(X2, Y)
        r_sq_enrollment_cergy2 = model.score(X2, Y)
        y_hat_p2_enrollment_cergy = model.predict(X2)

        X3 = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
        model = LinearRegression()
        model.fit(X3, Y)
        r_sq_enrollment_cergy3 = model.score(X3, Y)
        y_hat_p3_enrollment_cergy = model.predict(X3)

        X4 = PolynomialFeatures(degree=4, include_bias=False).fit_transform(X)
        model = LinearRegression()
        model.fit(X4, Y)
        r_sq_enrollment_cergy4 = model.score(X4, Y)
        y_hat_p4_enrollment_cergy = model.predict(X4)

        xaxis_enrollment_cergy = df_student_cergy['Course_year'].tolist()
        yaxis_enrollment_cergy = Y

        # *****************************END CERGY REGRESSION MODEL FOR AVERAGE ENROLLMENT END*************************************

        # ****************************NO OF STUDENT INTERNSHIPS FOR ENGINEERING PROGRAM EVERY YEAR ****************************
        test2 = pd.pivot_table(internship_df, index=["Year"], columns='Course', values="Student_id",
                               aggfunc=np.count_nonzero, fill_value=0)
        test22 = test2[["ING1", "ING2", "ING3"]].reset_index()

        # ***************************** LABELS *************************************

        y_1314_INTERN_labels = test22['Year'].tolist()

        # ***************************** COUNTS *************************************

        y_1314_INTERN_ING1 = test22['ING1'].tolist()
        y_1314_INTERN_ING2 = test22['ING2'].tolist()
        y_1314_INTERN_ING3 = test22['ING3'].tolist()

        # ****************************END NO OF STUDENT INTERNSHIPS FOR ENGINEERING PROGRAM EVERY YEAR END****************************

        # **************************** AVERAGE RENUMERATION FOR ALL THE ENGINEERNG COURSE EVERY YEAR *************************************

        test_R = pd.pivot_table(internship_df, index=["Year"], columns='Course', values="Pay_Details", aggfunc=np.mean, fill_value=0)
        test_new = test_R[["ING1", "ING2", "ING3"]].reset_index()

        # ***************************** LABELS *************************************

        y_1314_Renu_labels = test_new['Year'].tolist()

        # ***************************** COUNTS *************************************

        y_1314_ING1 = test_new['ING1'].tolist()
        y_1314_ING2 = test_new['ING2'].tolist()
        y_1314_ING3 = test_new['ING3'].tolist()

        # ****************************END AVERAGE RENUMERATION FOR ALL THE ENGINEERNG COURSE EVERY YEAR END *********************

        # ****************************ENROLLMENTS IN CAMPUSES IN ENGINEERING COURSES EVERY YEAR *********************

        sample_st = pd.pivot_table(college_df, index=["Course_year"], columns='Stream', values="Student_id", aggfunc=np.count_nonzero, fill_value=0)
        sample_stt = sample_st[["ING1", "ING2", "ING3"]].reset_index()

        # ***************************** LABELS *************************************

        y_1314_ENROLL_labels = sample_stt['Course_year'].tolist()

        # ***************************** COUNTS *************************************

        y_1314_ENROLL_ING1 = sample_stt['ING1'].tolist()
        y_1314_ENROLL_ING2 = sample_stt['ING2'].tolist()
        y_1314_ENROLL_ING3 = sample_stt['ING3'].tolist()

        # ****************************END ENROLLMENTS IN CAMPUSES IN ENGINEERING COURSES EVERY YEAR END*********************

        data = {
            "labels_pie": labels_pie,
            "count_of_students_pie": count_of_students_pie,
            "labels_bar": labels_bar,
            "count_of_students_bar": count_of_students_bar,
            "c_label": c_label,
            "c_data": c_data,
            "p_label": p_label,
            "p_data": p_data,
            "y_1314_program_labels": y_1314_program_labels,
            "y_1415_program_labels": y_1415_program_labels,
            "y_1516_program_labels": y_1516_program_labels,
            "y_1617_program_labels": y_1617_program_labels,
            "y_1718_program_labels": y_1718_program_labels,
            "y_1314_student_counts": y_1314_student_counts,
            "y_1415_student_counts": y_1415_student_counts,
            "y_1516_student_counts": y_1516_student_counts,
            "y_1617_student_counts": y_1617_student_counts,
            "y_1718_student_counts": y_1718_student_counts,
            "y_1314_program_labels_pau": y_1314_program_labels_pau,
            "y_1415_program_labels_pau": y_1415_program_labels_pau,
            "y_1516_program_labels_pau": y_1516_program_labels_pau,
            "y_1617_program_labels_pau": y_1617_program_labels_pau,
            "y_1718_program_labels_pau": y_1718_program_labels_pau,
            "y_1314_student_counts_pau": y_1314_student_counts_pau,
            "y_1415_student_counts_pau": y_1415_student_counts_pau,
            "y_1516_student_counts_pau": y_1516_student_counts_pau,
            "y_1617_student_counts_pau": y_1617_student_counts_pau,
            "y_1718_student_counts_pau": y_1718_student_counts_pau,
            "xaxis_pau" : xaxis_pau,
            "yaxis_pau" : yaxis_pau,
            "y_hat_p1_pau" : y_hat_p1_pau,
            "y_hat_p2_pau" : y_hat_p2_pau,
            "y_hat_p3_pau" : y_hat_p3_pau,
            "xaxis_cergy": xaxis_cergy,
            "yaxis_cergy": yaxis_cergy,
            "y_hat_p1_cergy": y_hat_pl_cergy,
            "y_hat_p2_cergy": y_hat_p2_cergy,
            "y_hat_p3_cergy": y_hat_p3_cergy,

            "xaxis_enrollment_pau": xaxis_enrollment_pau,
            "yaxis_enrollment_pau": yaxis_enrollment_pau,
            "y_hat_p1_enrollment_pau": y_hat_p1_enrollment_pau,
            "y_hat_p2_enrollment_pau": y_hat_p2_enrollment_pau,
            "y_hat_p3_enrollment_pau": y_hat_p3_enrollment_pau,
            "y_hat_p4_enrollment_pau": y_hat_p4_enrollment_pau,

            "xaxis_enrollment_cergy": xaxis_enrollment_cergy,
            "yaxis_enrollment_cergy": yaxis_enrollment_cergy,
            "y_hat_p1_enrollment_cergy": y_hat_p1_enrollment_cergy,
            "y_hat_p2_enrollment_cergy": y_hat_p2_enrollment_cergy,
            "y_hat_p3_enrollment_cergy": y_hat_p3_enrollment_cergy,
            "y_hat_p4_enrollment_cergy": y_hat_p4_enrollment_cergy,

            "y_1314_INTERN_labels": y_1314_INTERN_labels,
            "y_1314_INTERN_ING1": y_1314_INTERN_ING1,
            "y_1314_INTERN_ING2": y_1314_INTERN_ING2,
            "y_1314_INTERN_ING3": y_1314_INTERN_ING3,

            "y_1314_ENROLL_labels": y_1314_ENROLL_labels,
            "y_1314_ENROLL_ING1": y_1314_ENROLL_ING1,
            "y_1314_ENROLL_ING2": y_1314_ENROLL_ING2,
            "y_1314_ENROLL_ING3": y_1314_ENROLL_ING3,

            "y_1314_Renu_labels": y_1314_Renu_labels,
            "y_1314_ING1": y_1314_ING1,
            "y_1314_ING2": y_1314_ING2,
            "y_1314_ING3": y_1314_ING3,


            "dist": dist,
            "dist1": dist1,
            "dist2": dist2,

        }
        return Response(data)


class ChartFilterData(APIView):
    authentication_classes = ()
    permission_classes = ()

    def get(self, request, format=None):

        # *****************************LOADING DATA FROM MODELS*************************************

        internship_table = str(Internship.objects.all().query)
        internship_df = pd.read_sql_query(internship_table, connection)

        # *****************************END DATA FROM MODELS END *************************************

        # ***************************** TOP 10 Recruiters Every Year START*************************************

        temp_1314 = internship_df.loc[(internship_df['Year'] == '2013/2014')]
        y_1314 = temp_1314['Company_Name'].value_counts().rename_axis('company').head(10).reset_index(name='count')

        temp_1415 = internship_df.loc[(internship_df['Year'] == '2014/2015')]
        y_1415 = temp_1415['Company_Name'].value_counts().rename_axis('company').head(10).reset_index(name='count')

        temp_1516 = internship_df.loc[(internship_df['Year'] == '2015/2016')]
        y_1516 = temp_1516['Company_Name'].value_counts().rename_axis('company').head(10).reset_index(name='count')

        temp_1617 = internship_df.loc[(internship_df['Year'] == '2016/2017')]
        y_1617 = temp_1617['Company_Name'].value_counts().rename_axis('company').head(10).reset_index(name='count')

        temp_1718 = internship_df.loc[(internship_df['Year'] == '2017/2018')]
        y_1718 = temp_1718['Company_Name'].value_counts().rename_axis('company').head(10).reset_index(name='count')

        # ***************************** LABELS *************************************

        y_1314_company_labels = y_1314['company'].tolist()
        y_1415_company_labels = y_1415['company'].tolist()
        y_1516_company_labels = y_1516['company'].tolist()
        y_1617_company_labels = y_1617['company'].tolist()
        y_1718_company_labels = y_1718['company'].tolist()

        # ***************************** COUNTS *************************************

        y_1314_company_counts = y_1314['count'].tolist()
        y_1415_company_counts = y_1415['count'].tolist()
        y_1516_company_counts = y_1516['count'].tolist()
        y_1617_company_counts= y_1617['count'].tolist()
        y_1718_company_counts = y_1718['count'].tolist()

        # *****************************END TOP 10 Recruiters Every Year END*************************************

        data = {
            "y_1314_company_labels" : y_1314_company_labels,
            "y_1415_company_labels" : y_1415_company_labels,
            "y_1516_company_labels" : y_1516_company_labels,
            "y_1617_company_labels" : y_1617_company_labels,
            "y_1718_company_labels" : y_1718_company_labels,
            "y_1314_company_counts" : y_1314_company_counts,
            "y_1415_company_counts" : y_1415_company_counts,
            "y_1516_company_counts" : y_1516_company_counts,
            "y_1617_company_counts" : y_1617_company_counts,
            "y_1718_company_counts" : y_1718_company_counts,
        }
        return Response(data)
