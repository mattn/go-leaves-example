package main

import (
	"fmt"
	"log"

	"github.com/dmitryikh/leaves"
	"github.com/dmitryikh/leaves/mat"
	"github.com/dmitryikh/leaves/util"
)

func main() {
	test, err := mat.DenseMatFromCsvFile("iris_test.tsv", 0, false, "\t", 0.0)
	if err != nil {
		log.Fatal(err)
	}

	model, err := leaves.LGEnsembleFromFile("lg_iris.model", true)
	if err != nil {
		log.Fatal(err)
	}

	predictions := make([]float64, test.Rows*model.NOutputGroups())
	model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)

	truePredictions, err := mat.DenseMatFromCsvFile("iris_pred.tsv", 0, false, "\t", 0.0)
	if err != nil {
		panic(err)
	}

	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, 1e-6); err != nil {
		panic(fmt.Errorf("different raw predictions: %s", err.Error()))
	}
}
