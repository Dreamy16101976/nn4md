/*  nn4md - neural net for pulse induction metal detector
    Copyright (C) 2019 Alexey "FoxyLab" Voronin
    Email:    support@foxylab.com
    Website:  https://acdc.foxylab.com

	This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

*/

package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/fatih/color"
)

const (
	inputs         = 8            //number of input nodes
	defaultHiddens = 3            //number of hidden nodes
	outputs        = 2            //number of output nodes
	weightStart    = 0.1          //start weights value
	defaultα       = 0.1          //default learning rate
	errThreshold   = 0.01         //MSE threshold
	learningSize   = 110          //train data size
	testSize       = 40           //validation data size
	scaleIn        = 1024         //input data scaling factor
	scaleOut       = 1.0          //output data scaling factor
	trainFileName  = "train.dat"  //train data filename
	validFileName  = "test.dat"   //validation data filename
	jsonFileName   = "nn4md.json" //JSON filename
)

//error check
func check(e error) {
	if e != nil {
		panic(e)
	}
}

//matrix initialization
func mat2D(rows, cols int) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, cols)
	}
	return mat
}

func mat1D(rows int, value float64) []float64 {
	mat := make([]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = value
	}
	return mat
}

//rnd generation
func rnd(a, b float64) float64 {
	return a + (b-a)*rand.Float64()
}

//activation functions
func logistic(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//activation functions derivatives
func dLogistic(x float64) float64 {
	return x * (1 - x)
}

//JSON
type Layer struct {
	Type       string      `json:"type"`
	Activation string      `json:"activation,omitempty"`
	Neurons    int         `json:"neurons,omitempty"`
	Weights    [][]float64 `json:"weights,omitempty"`
}

type JSONstruct struct {
	Layers []Layer `json:"layers"`
}

var (
	inAct, hidAct, outAct  []float64   //nodes activations
	hidWeights, outWeights [][]float64 //nodes weights
	seedString             string
	wSeed                  int64
	hiddensString          string
	hiddens64              int64
	hiddens                int
	αString                string
	α                      float64
	err                    error
)

//net init
func build(inputs, hiddens, outputs int) {
	inAct = mat1D(inputs+1, 1.0)
	hidAct = mat1D(hiddens+1, 1.0)
	outAct = mat1D(outputs, 1.0)
	hidWeights = mat2D(inputs+1, hiddens)
	outWeights = mat2D(hiddens+1, outputs)
	for i := 0; i < inputs+1; i++ {
		for j := 0; j < hiddens; j++ {
			hidWeights[i][j] = rnd(-weightStart, weightStart)
		}
	}
	for i := 0; i < hiddens+1; i++ {
		for j := 0; j < outputs; j++ {
			outWeights[i][j] = rnd(-weightStart, weightStart)
		}
	}
}

//net solve
func guess(inputLayer []float64) []float64 {
	for i := 0; i < inputs; i++ {
		//input neuron output
		inAct[i] = inputLayer[i]
	}
	for i := 0; i < hiddens; i++ {
		//hidden neuron state
		Σ := 0.0
		for j := 0; j < inputs+1; j++ {
			Σ += inAct[j] * hidWeights[j][i]
		}
		//hidden neuron output
		hidAct[i] = logistic(Σ)
	}
	for i := 0; i < outputs; i++ {
		//output neuron state
		Σ := 0.0
		for j := 0; j < hiddens+1; j++ {
			Σ += hidAct[j] * outWeights[j][i]
		}
		//output neuron output
		outAct[i] = logistic(Σ)
	}
	//return output neuron outputs
	return outAct
}

//net learn
func learn(targets []float64, α float64) float64 {
	//output layer weights update
	outDeltas := mat1D(outputs, 0.0)
	for i := 0; i < hiddens+1; i++ {
		for j := 0; j < outputs; j++ {
			outDeltas[j] = dLogistic(outAct[j]) * (outAct[j] - targets[j])
			//              ∂yout_j / ∂zout_j   *        ∂e/dyout_j
			outWeights[i][j] = outWeights[i][j] - α*(outDeltas[j]*hidAct[i]) //update rule
			//                                                    * ∂zout_j / ∂wi
			//                                    for bias          = 1.0
		}
	}
	//hidden layer weights update
	hidDeltas := mat1D(hiddens, 0.0)
	for i := 0; i < hiddens; i++ {
		Σ := 0.0
		for j := 0; j < outputs; j++ {
			Σ += outDeltas[j] * outWeights[i][j]

		}
		// -> ∂e/∂yh_i
		hidDeltas[i] = dLogistic(hidAct[i]) * Σ
		//             ∂yh_i / ∂w_i         * ∂e/∂yh_i
	}
	for i := 0; i < inputs+1; i++ {
		for j := 0; j < hiddens; j++ {
			hidWeights[i][j] = hidWeights[i][j] - α*(hidDeltas[j]*inAct[i]) //update rule
			//                                                    * ∂zh_i / ∂w_i
			//                                    for bias          =  1.0
		}
	}
	sse := 0.0
	for i := 0; i < outputs; i++ {
		sse += math.Pow(targets[i]-outAct[i], 2)
	}
	return sse
}

func main() {
	//parameters reading
	flag.StringVar(&seedString, "s", "", "Seed")              //seed
	flag.StringVar(&hiddensString, "h", "", "Hidden Neurons") //hiddens
	flag.StringVar(&αString, "r", "", "Learning Rate")        //learning rate
	flag.Parse()
	wSeed = 0
	//get seed
	if seedString != "" {
		wSeed, err = strconv.ParseInt(seedString, 10, 0)
		check(err)
	}
	fmt.Println("Seed: ", wSeed)
	hiddens = defaultHiddens
	//get hiddens
	if hiddensString != "" {
		hiddens64, err = strconv.ParseInt(hiddensString, 10, 0)
		check(err)
		hiddens = int(hiddens64)
	}
	fmt.Println("Hidden Neurons: ", hiddens)
	α = defaultα
	//get learning rate
	if αString != "" {
		α, err = strconv.ParseFloat(αString, 64)
		check(err)
	}
	fmt.Println("Learning Rate: ", α)
	//initialize NN
	rand.Seed(wSeed)
	build(inputs, hiddens, outputs)
	fmt.Println("Parse train data...")
	patterns := make([][][]float64, learningSize)
	for i := 0; i < learningSize; i++ {
		patterns[i] = make([][]float64, 2)
		patterns[i][0] = make([]float64, inputs)
		patterns[i][1] = make([]float64, outputs)
	}
	learningData, err := os.Open(trainFileName)
	check(err)
	defer learningData.Close()
	reader := bufio.NewReader(learningData)
	var line string
	for i := 0; i < learningSize; i++ {
		line, err = reader.ReadString('\n')
		value := ""
		idx := 0
		for j := 0; j < len(line); j++ {
			if line[j] == 0x0a {
				patterns[i][1][idx-inputs], err = strconv.ParseFloat(value, 64)
				patterns[i][1][idx-inputs] = patterns[i][1][idx-inputs] / scaleOut
				check(err)
				break
			}
			if line[j] == 0x09 {
				if idx > (inputs - 1) {
					patterns[i][1][idx-inputs], err = strconv.ParseFloat(value, 64)
					patterns[i][1][idx-inputs] = patterns[i][1][idx-inputs] / scaleOut
					check(err)
				} else {
					patterns[i][0][idx], err = strconv.ParseFloat(value, 64)
					patterns[i][0][idx] = patterns[i][0][idx] / scaleIn
					check(err)
				}
				idx++
				value = ""
			} else {
				value = value + string(line[j])
			}
		}
		if err != nil {
			break
		}
	}
	//shuffle learning data
	fmt.Println("Shuffle...")
	rand.Seed(7777)
	for i := learningSize - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		patterns[i], patterns[j] = patterns[j], patterns[i]
	}
	fmt.Println("Parse validation data...")
	validationData := make([][][]float64, testSize)
	for i := 0; i < testSize; i++ {
		validationData[i] = make([][]float64, 2)
		validationData[i][0] = make([]float64, inputs)
		validationData[i][1] = make([]float64, outputs)
	}
	testData, err := os.Open(validFileName)
	check(err)
	defer testData.Close()
	reader = bufio.NewReader(testData)
	for i := 0; i < testSize; i++ {
		line, err = reader.ReadString('\n')
		value := ""
		idx := 0
		for j := 0; j < len(line); j++ {
			if line[j] == 0x0a {
				validationData[i][1][idx-inputs], err = strconv.ParseFloat(value, 64)
				validationData[i][1][idx-inputs] = validationData[i][1][idx-inputs] / scaleOut
				check(err)
				break
			}
			if line[j] == 0x09 {
				if idx > (inputs + 1 - 2) {
					validationData[i][1][idx-inputs], err = strconv.ParseFloat(value, 64)
					validationData[i][1][idx-inputs] = validationData[i][1][idx-inputs] / scaleOut
					check(err)
				} else {
					validationData[i][0][idx], err = strconv.ParseFloat(value, 64)
					validationData[i][0][idx] = validationData[i][0][idx] / scaleIn
					check(err)
				}
				idx++
				value = ""
			} else {
				value = value + string(line[j])
			}
		}
		if err != nil {
			break
		}
	}
	//training
	fmt.Println("--- TRAINING ---")
	epoch := 0
	iteration := 0
	lSSE := 0.0
	lMSE := 0.0
	tSSE := 0.0 
	tMSE := 0.0
	ok := 0
	rand.Seed(8888)
	for {
		//train
		guess(patterns[iteration][0])
		lSSE += learn(patterns[iteration][1], α)
		//validation
		iteration++
		if iteration == learningSize {
			//epoch
			epoch++
			lMSE = lSSE / float64(learningSize)
			fmt.Print("Epoch: ", epoch)
			fmt.Printf("\tMSE: %.5f", lMSE)
			ok = 0
			tSSE = 0.0
			for _, p := range validationData {
				ansNN := 0
				max := 0.0
				for outputIdx := 0; outputIdx < outputs; outputIdx++ {
					tSSE += math.Pow(guess(p[0])[outputIdx]-p[1][outputIdx], 2)
				}
				for outputIdx := 0; outputIdx < outputs; outputIdx++ {
					if (guess(p[0]))[outputIdx] > max {
						max = guess(p[0])[outputIdx]
						ansNN = outputIdx
					}
				}
				ansTest := 0
				max = 0.0
				for outputIdx := 0; outputIdx < outputs; outputIdx++ {
					if p[1][outputIdx] > max {
						max = p[1][outputIdx]
						ansTest = outputIdx
					}
				}
				if ansNN == ansTest {
					ok++
				}
			}
			tMSE = tSSE / testSize
			fmt.Printf("\tMSE: %.5f", tMSE)
			fmt.Print("\tAcc.: ", ok)
			fmt.Printf("\t%.2f", float64(ok)/float64(testSize)*100.0)
			fmt.Println(" %")
			if tMSE < errThreshold {
				break
			}
			iteration = 0
			lSSE = 0.0
		}
	}
	fmt.Println("Test results:")
	for _, p := range validationData {
		ansNN := 0
		max := 0.0
		for outputIdx := 0; outputIdx < outputs; outputIdx++ {
			if (guess(p[0]))[outputIdx] > max {
				max = guess(p[0])[outputIdx]
				ansNN = outputIdx
			}
		}
		ansTest := 0
		max = 0.0
		for outputIdx := 0; outputIdx < outputs; outputIdx++ {
			if p[1][outputIdx] > max {
				max = p[1][outputIdx]
				ansTest = outputIdx
			}
		}
		if ansNN == ansTest {
			color.Set(color.FgGreen)
		} else {
			color.Set(color.FgRed)
		}
		fmt.Println(ansTest, " -> ", ansNN)
		color.Unset()
	}
	fmt.Print("Epoch: ", epoch)
	fmt.Printf("\tMSE: %.5f", lMSE)
	fmt.Printf("\tMSE: %.5f", tMSE)
	fmt.Print("\tAcc.: ", ok)
	fmt.Printf("\t%.2f", float64(ok)/float64(testSize)*100.0)
	fmt.Println(" %")
	//saving weights to JSON file
	var jsonStruct JSONstruct
	jsonStruct.Layers = make([]Layer, 3)
	jsonStruct.Layers[0] = Layer{Type: "input", Activation: "linear", Neurons: inputs}
	jsonStruct.Layers[1] = Layer{Type: "hidden", Activation: "logistic", Neurons: hiddens, Weights: hidWeights}
	jsonStruct.Layers[2] = Layer{Type: "output", Activation: "logistic", Neurons: outputs, Weights: outWeights}
	jsonData, err := json.Marshal(jsonStruct)
	jsonFile, err := os.Create(jsonFileName)
	check(err)
	defer jsonFile.Close()
	jsonFile.Write(jsonData)
	testSet := make([]float64, inputs)
	fmt.Println("--- TESTING ---")
	for {
		fmt.Println("Input test data:")
		for i := 0; i < inputs; i++ {
			var inp float64
			fmt.Print(i+1, ":")
			fmt.Scanln(&inp)
			testSet[i] = inp / scaleIn
		}
		fmt.Println("Outputs:")
		fmt.Println(guess(testSet))
		ansNN := 0
		max := 0.0
		for outputIdx := 0; outputIdx < outputs; outputIdx++ {
			if (guess(testSet))[outputIdx] > max {
				max = (guess(testSet))[outputIdx]
				ansNN = outputIdx
			}
		}
		fmt.Println("Answer: ", ansNN)
	}
}
