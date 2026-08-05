package tf_test

import (
	"fmt"

	"github.com/KEINOS/go-nn/nn"
	"github.com/KEINOS/go-tf/tf"
)

func ExampleNewModel() {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		fmt.Println("backend error:", err)
		return
	}
	defer backend.Close()

	model, err := tf.NewModel(backend, 4, 3, nn.NewGenerator(42, 0))
	if err != nil {
		fmt.Println("model error:", err)
		return
	}
	defer model.Close()

	output, err := model.Forward([]int{2, 1}, 2)
	if err != nil {
		fmt.Println("forward error:", err)
		return
	}

	fmt.Printf("parameters=%d\n", len(model.Parameters()))
	fmt.Printf("output_shape=%v\n", output.Value().Shape())
	// Output:
	// parameters=1
	// output_shape=[2 3]
}
