from django.urls import path

from .views import BlogList, BlogDetailView, AboutPageView, SciencePageView, EntertainmentPageView, \
    NeanderthalPageView,  logout_user, LSTM_n, LSTM_site_view, SearchResultsView, BigSearchResultsView,Gradient_site_view,Gradient,RNN_site_view,RNN, LSTM2, LSTM2_site_view
from .views import BlogList, BlogDetailView, AboutPageView, SciencePageView, EntertainmentPageView, NeanderthalPageView


urlpatterns = [
    path('', BlogList.as_view(), name='home'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('post/<int:pk>/', BlogDetailView.as_view(), name='post_detail'),
    path('science/', SciencePageView.as_view(), name='science'),
    path('entertainment/', EntertainmentPageView.as_view(), name='entertainment'),
    path('neanderthal/', NeanderthalPageView.as_view(), name='neanderthal'),
    path('logout/', logout_user, name='logout'),
    path('LSTM/', LSTM_site_view, name='LSTM'),
    path('restart_pol_reg/', LSTM_n, name='restart_pol_reg'),
    path('restart_gradient/',Gradient,name='restart_gradient'),
    path('search/', SearchResultsView.as_view(), name='search'),
    path('big_search/', BigSearchResultsView.as_view(), name='big_search'),
    path('Gradient/',Gradient_site_view, name='Gradient'),
    path('RNN/',RNN_site_view,name='RNN'),
    path('LSTM2/',LSTM2_site_view,name='LSTM2')
]


