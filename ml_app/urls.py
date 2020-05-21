from django.urls import path, include
from . import views, views_alternative, views_canvas
from rest_framework import routers

router = routers.DefaultRouter()
router.register('ml_app', views.ApprovalsView)

urlpatterns = [
    path('yzat/', views_alternative.credit, name='anasayfa'),
    path('api/', include(router.urls)),
    path('status/', views_alternative.approvereject),
    path('creditai', views.cxcontact, name='cxform'),

]