nn4md
=====================

Neural net for pulse induction metal detector / Нейронная сеть для импульсного металлодетектора<br/>
<br/>
(C) 2019 Alexey "FoxyLab" Voronin<br/>E-mail: support@foxylab.com<br/>WWW: https://acdc.foxylab.com<br/>
<br/>
A neural network that processes measurement results and discriminates targets.<br/>
<br/>
Build:
===
go get -u 
go build nn4md.go
<br/>
Use:
===
nn4md [-s seed] [-h hiddens] [-r rate]
seed - seed for PRNG
hiddens - number of hidden layer neurons
rate - learning rate
***
Нейронная сеть, обрабатывающая результаты измерений и выполняющая дискриминацию мишеней.<br/>
