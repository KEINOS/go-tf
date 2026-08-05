package tf

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/KEINOS/go-nn/nn"
)

func TestModelUsesGoNNParameterState(t *testing.T) {
	t.Parallel()

	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(backend.Close)

	model, err := NewModel(backend, 3, 2, nn.NewGenerator(11, 0))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(model.Close)

	parameters := model.Parameters()
	if len(parameters) != 1 {
		t.Fatalf("parameter count = %d, want 1", len(parameters))
	}

	if parameters[0].Name() != "token.embedding.weight" {
		t.Fatalf("parameter name = %q", parameters[0].Name())
	}

	state, err := model.StateDict()
	if err != nil {
		t.Fatal(err)
	}

	shape := state.Entries()[0].Shape
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 2 {
		t.Fatalf("shape = %v, want [3 2]", shape)
	}
}

func TestModelForwardConsumesTensorEmbeddingPrimitive(t *testing.T) {
	t.Parallel()

	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(backend.Close)
	model, err := NewModel(backend, 3, 2, nn.NewGenerator(12, 0))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(model.Close)

	output, err := model.Forward([]int{2, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got := output.Value().Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("shape = %v, want [2 2]", got)
	}

	loss, err := output.Sum()
	if err != nil {
		t.Fatal(err)
	}
	if err = loss.Backward(); err != nil {
		t.Fatal(err)
	}
	optimizer, err := model.NewOptimizer(backend, nn.NewTensorAdamConfig())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(optimizer.Close)
	if err = optimizer.Step(); err != nil {
		t.Fatal(err)
	}
	if optimizer.StepCount() != 1 {
		t.Fatalf("optimizer step = %d, want 1", optimizer.StepCount())
	}
}

func TestModelTrainingCheckpointRoundTrip(t *testing.T) {
	t.Parallel()

	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(backend.Close)
	generator := nn.NewGenerator(77, 3)
	model, err := NewModel(backend, 3, 2, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(model.Close)
	optimizer, err := model.NewOptimizer(backend, nn.NewTensorAdamConfig())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(optimizer.Close)

	output, err := model.Forward([]int{0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := output.Sum()
	if err != nil {
		t.Fatal(err)
	}
	if err = loss.Backward(); err != nil {
		t.Fatal(err)
	}
	if err = optimizer.Step(); err != nil {
		t.Fatal(err)
	}
	context, err := nn.NewExecutionContext(nn.Training, generator)
	if err != nil {
		t.Fatal(err)
	}

	wantModel, err := model.StateDict()
	if err != nil {
		t.Fatal(err)
	}
	wantOptimizer, err := optimizer.State()
	if err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err = model.WriteTrainingCheckpoint(&encoded, optimizer, context); err != nil {
		t.Fatal(err)
	}
	restored, err := model.RestoreTrainingCheckpoint(
		bytes.NewReader(encoded.Bytes()), backend, optimizer, nn.DefaultCheckpointLimits(),
	)
	if err != nil {
		t.Fatal(err)
	}
	gotModel, err := model.StateDict()
	if err != nil {
		t.Fatal(err)
	}
	gotOptimizer, err := optimizer.State()
	if err != nil {
		t.Fatal(err)
	}
	if !equalStateEntries(wantModel.Entries(), gotModel.Entries()) ||
		!equalOptimizerState(wantOptimizer, gotOptimizer) {
		t.Fatal("training checkpoint did not reproduce model and optimizer state")
	}
	if restored.Mode() != nn.Training {
		t.Fatalf("restored mode = %v, want training", restored.Mode())
	}
}

func equalStateEntries(left, right []nn.StateEntry) bool {
	return reflect.DeepEqual(left, right)
}

func equalOptimizerState(left, right nn.TensorAdamState) bool {
	return reflect.DeepEqual(left, right)
}
