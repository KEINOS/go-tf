package tf_test

import (
	"bytes"
	"fmt"

	"github.com/KEINOS/go-nn/nn"
	"github.com/KEINOS/go-tf/tf"
)

func ExampleModel_forwardShape() {
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

	fmt.Printf("output_shape=%v\n", output.Value().Shape())
	// Output:
	// output_shape=[2 3]
}

func ExampleModel_stateDictRoundTrip() {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		fmt.Println("backend error:", err)
		return
	}
	defer backend.Close()

	source, err := tf.NewModel(backend, 4, 3, nn.NewGenerator(1, 0))
	if err != nil {
		fmt.Println("source error:", err)
		return
	}
	defer source.Close()

	target, err := tf.NewModel(backend, 4, 3, nn.NewGenerator(2, 0))
	if err != nil {
		fmt.Println("target error:", err)
		return
	}
	defer target.Close()

	state, err := source.StateDict()
	if err != nil {
		fmt.Println("state error:", err)
		return
	}

	if err = target.LoadStateDict(state); err != nil {
		fmt.Println("load error:", err)
		return
	}

	loaded, err := target.StateDict()
	if err != nil {
		fmt.Println("loaded error:", err)
		return
	}

	fmt.Printf("entries=%d\n", len(loaded.Entries()))
	fmt.Printf("shape=%v\n", loaded.Entries()[0].Shape)
	// Output:
	// entries=1
	// shape=[4 3]
}

func ExampleModel_trainingCheckpointRoundTrip() {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		fmt.Println("backend error:", err)
		return
	}
	defer backend.Close()

	generator := nn.NewGenerator(77, 3)
	model, err := tf.NewModel(backend, 3, 2, generator)
	if err != nil {
		fmt.Println("model error:", err)
		return
	}
	defer model.Close()

	optimizer, err := model.NewOptimizer(backend, nn.NewTensorAdamConfig())
	if err != nil {
		fmt.Println("optimizer error:", err)
		return
	}
	defer optimizer.Close()

	output, err := model.Forward([]int{0}, 1)
	if err != nil {
		fmt.Println("forward error:", err)
		return
	}

	loss, err := output.Sum()
	if err != nil {
		fmt.Println("sum error:", err)
		return
	}

	if err = loss.Backward(); err != nil {
		fmt.Println("backward error:", err)
		return
	}

	if err = optimizer.Step(); err != nil {
		fmt.Println("step error:", err)
		return
	}

	context, err := nn.NewExecutionContext(nn.Training, generator)
	if err != nil {
		fmt.Println("context error:", err)
		return
	}

	var encoded bytes.Buffer
	if err = model.WriteTrainingCheckpoint(&encoded, optimizer, context); err != nil {
		fmt.Println("write error:", err)
		return
	}

	restored, err := model.RestoreTrainingCheckpoint(
		bytes.NewReader(encoded.Bytes()),
		backend,
		optimizer,
		nn.DefaultCheckpointLimits(),
	)
	if err != nil {
		fmt.Println("restore error:", err)
		return
	}

	fmt.Printf("training=%v\n", restored.Mode() == nn.Training)
	fmt.Printf("step_count=%d\n", optimizer.StepCount())
	// Output:
	// training=true
	// step_count=1
}
