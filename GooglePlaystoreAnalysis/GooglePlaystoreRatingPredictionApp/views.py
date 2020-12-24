# Create your views here.
# Importing Libraries
from django.shortcuts import render
from django.conf import settings
import os
import pickle
import numpy as np
from scipy import integrate
from sklearn.preprocessing import LabelEncoder


def GooglePlaystoreRatingPrediction(request):
    d = True
    ans = True
    categories = pickle.load(open('categories.pkl', 'rb'))
    content_ratings = pickle.load(open('content_ratings.pkl', 'rb'))

    if request.method == "POST":

        a = {}
        print(type(a))

        print(request.POST)
        print(len(request.POST))
        last_idx = len(request.POST) - 1

        for key, value in request.POST.items():
            a[key] = value
        del a['csrfmiddlewaretoken']
        del a['Submit']
        print(a)

        # Changing datatypes according to dataset

        # Load saved objects and pretrained model's pickle file for Label Encoding and Model Prediction
        type_le = pickle.load(open('type_le.pkl', 'rb'))
        content_le = pickle.load(open('content_le.pkl', 'rb'))
        category_le = pickle.load(open('category_le.pkl', 'rb'))
        dt_reg = pickle.load(open('decision_tree_regressor.pkl', 'rb'))

        a['Category'] = category_le.transform([a['Category']])[0]
        a['Type'] = type_le.transform([a['Type']])[0]
        a['Content Rating'] = content_le.transform([a['Content Rating']])[0]
        a['Price'] = float(a['Price'])
        cols = ['Reviews', 'Size', 'Installs']
        for col in cols:
            a[col] = np.log(float(a[col]))
            print("{}: {}".format(col, a[col]))
        print(a)

        given_details = []
        for k, v in a.items():
            given_details.append(v)
        print("Final Preprocessed Data: ", given_details)

        # Predicting Rating
        y_pred = dt_reg.predict([given_details])[0]
        ans = round(np.exp(y_pred), 2)
        print(ans)
        d = False
    return render(request, 'rating_prediction.html', {'d': d, 'ans': ans, 'categories': categories, 'content_ratings': content_ratings})
