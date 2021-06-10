from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
from .apps import WandmindConfig

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import WandmindConfig

class call_model(APIView):

    def get(self,request):
        if request.method == 'GET':
            
            # sentence is the query we want to get the prediction for
            params =  request.GET.get('poetry_number')
            
            # predict method used to get the prediction
            response = WandmindConfig.predictor(params)
            
            # returning JSON response
            return HttpResponse(response)
            
    
def HomePage(request):
    return render(request, 'home.html')

def SearchPage(request):
    #return render(request, ('model/?number='+request))
    params =  request.GET.get('poetry_number')
            
    # predict method used to get the prediction
    response = WandmindConfig.predictor(params)
            
    # returning JSON response
    return HttpResponse(response)

