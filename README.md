# nn4md
Neural net for pulse induction metal detector<br/>
<br/>
(C) 2019 *Alexey "FoxyLab" Voronin*<br/>
E-mail: support@foxylab.com<br/>
WWW: https://acdc.foxylab.com<br/>
<br/>
A neural network that processes measurement results and discriminates targets.<br/>
<br/>
## Build
`go get github.com/fatih/color`<br/>
`go build nn4md.go`
<br/>
## Use
`nn4md [-s seed] [-h hiddens] [-r rate]`<br/>
*seed* - seed for PRNG<br/>
*hiddens* - number of hidden layer neurons<br/>
*rate* - learning rate<br/>
**Input**:<br/>
Train data file : *train.dat*<br/>
Validation data file : *test.dat*<br/>
**Output**:<br/>
Neural network structure file: *nn4md.json*<br/>
