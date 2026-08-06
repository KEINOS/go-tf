package tf

import (
	"bytes"
	"errors"
	"testing"

	"github.com/KEINOS/go-nn/nn"
	"github.com/KEINOS/go-nn/nn/nnerr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelUsesGoNNParameterState(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	model, err := NewModel(backend, 3, 2, nn.NewGenerator(11, 0))
	require.NoError(t, err)
	t.Cleanup(model.Close)

	parameters := model.Parameters()
	require.Len(t, parameters, 1)
	assert.Equal(t, "token.embedding.weight", parameters[0].Name())

	state, err := model.StateDict()
	require.NoError(t, err)

	shape := state.Entries()[0].Shape
	require.Len(t, shape, 2)
	assert.Equal(t, 3, shape[0])
	assert.Equal(t, 2, shape[1])
}

func TestModelParametersReturnsCallerOwnedSlice(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	model, err := NewModel(backend, 3, 2, nn.NewGenerator(11, 0))
	require.NoError(t, err)
	t.Cleanup(model.Close)

	parameters := model.Parameters()
	require.Len(t, parameters, 1)
	want := parameters[0]

	parameters[0] = nil
	parameters = append(parameters, nil)
	require.Len(t, parameters, 2)

	got := model.Parameters()
	require.Len(t, got, 1)
	require.Same(t, want, got[0])
	assert.Equal(t, "token.embedding.weight", got[0].Name())
}

func TestModelStateDictEntriesReturnCallerOwnedCopies(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	model, err := NewModel(backend, 3, 2, nn.NewGenerator(11, 0))
	require.NoError(t, err)
	t.Cleanup(model.Close)

	state, err := model.StateDict()
	require.NoError(t, err)

	entries := state.Entries()
	require.Len(t, entries, 1)
	require.NotEmpty(t, entries[0].Shape)
	require.NotEmpty(t, entries[0].Data)

	wantName := entries[0].Name
	wantShape := append([]int(nil), entries[0].Shape...)
	wantData := append([]float64(nil), entries[0].Data...)

	entries[0].Name = "corrupted.name"
	entries[0].Shape[0] = 99
	entries[0].Data[0] = 99

	gotSameState := state.Entries()
	require.Len(t, gotSameState, 1)
	assert.Equal(t, wantName, gotSameState[0].Name)
	assert.Equal(t, wantShape, gotSameState[0].Shape)
	assert.Equal(t, wantData, gotSameState[0].Data)

	freshState, err := model.StateDict()
	require.NoError(t, err)
	gotFresh := freshState.Entries()
	require.Len(t, gotFresh, 1)
	assert.Equal(t, wantName, gotFresh[0].Name)
	assert.Equal(t, wantShape, gotFresh[0].Shape)
	assert.Equal(t, wantData, gotFresh[0].Data)
}

func TestModelForwardConsumesTensorEmbeddingPrimitive(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)
	model, err := NewModel(backend, 3, 2, nn.NewGenerator(12, 0))
	require.NoError(t, err)
	t.Cleanup(model.Close)

	output, err := model.Forward([]int{2, 0}, 2)
	require.NoError(t, err)
	got := output.Value().Shape()
	require.Len(t, got, 2)
	assert.Equal(t, 2, got[0])
	assert.Equal(t, 2, got[1])

	loss, err := output.Sum()
	require.NoError(t, err)
	require.NoError(t, loss.Backward())
	optimizer, err := model.NewOptimizer(backend, nn.NewTensorAdamConfig())
	require.NoError(t, err)
	t.Cleanup(optimizer.Close)
	require.NoError(t, optimizer.Step())
	assert.EqualValues(t, 1, optimizer.StepCount())
}

func TestModelTrainingCheckpointRoundTrip(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)
	generator := nn.NewGenerator(77, 3)
	model, err := NewModel(backend, 3, 2, generator)
	require.NoError(t, err)
	t.Cleanup(model.Close)
	optimizer, err := model.NewOptimizer(backend, nn.NewTensorAdamConfig())
	require.NoError(t, err)
	t.Cleanup(optimizer.Close)

	output, err := model.Forward([]int{0}, 1)
	require.NoError(t, err)
	loss, err := output.Sum()
	require.NoError(t, err)
	require.NoError(t, loss.Backward())
	require.NoError(t, optimizer.Step())
	context, err := nn.NewExecutionContext(nn.Training, generator)
	require.NoError(t, err)

	wantModel, err := model.StateDict()
	require.NoError(t, err)
	wantOptimizer, err := optimizer.State()
	require.NoError(t, err)
	var encoded bytes.Buffer
	require.NoError(t, model.WriteTrainingCheckpoint(&encoded, optimizer, context))
	restored, err := model.RestoreTrainingCheckpoint(
		bytes.NewReader(encoded.Bytes()), backend, optimizer, nn.DefaultCheckpointLimits(),
	)
	require.NoError(t, err)
	gotModel, err := model.StateDict()
	require.NoError(t, err)
	gotOptimizer, err := optimizer.State()
	require.NoError(t, err)
	assert.Equal(t, wantModel.Entries(), gotModel.Entries())
	assert.Equal(t, wantOptimizer, gotOptimizer)
	assert.Equal(t, nn.Training, restored.Mode())
}

func TestModelLoadStateDict(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	model1, err := NewModel(backend, 4, 3, nn.NewGenerator(1, 0))
	require.NoError(t, err)
	t.Cleanup(model1.Close)

	model2, err := NewModel(backend, 4, 3, nn.NewGenerator(2, 0))
	require.NoError(t, err)
	t.Cleanup(model2.Close)

	state1, err := model1.StateDict()
	require.NoError(t, err)

	require.NoError(t, model2.LoadStateDict(state1))

	state2, err := model2.StateDict()
	require.NoError(t, err)
	assert.Equal(t, state1.Entries(), state2.Entries())
}

func TestNewModelRejectsInvalidDimensions(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	_, err = NewModel(backend, 0, 2, nn.NewGenerator(1, 0))
	require.ErrorIs(t, err, nnerr.ErrInvalidDimension)

	_, err = NewModel(backend, 2, 0, nn.NewGenerator(1, 0))
	require.ErrorIs(t, err, nnerr.ErrInvalidDimension)
}

func TestNewModelWrapsModuleCollectionError(t *testing.T) {
	original := newModuleCollection
	t.Cleanup(func() {
		newModuleCollection = original
	})

	sentinel := errors.New("module collection boom")
	newModuleCollection = func(*nn.TensorBackend) (*nn.ModuleCollection, error) {
		return nil, sentinel
	}

	_, err := NewModel(nil, 3, 2, nn.NewGenerator(1, 0))
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel)
	assert.ErrorContains(t, err, "create model modules")
}

func TestNewModelWrapsEmbeddingCreationError(t *testing.T) {
	original := newTensorEmbedding
	t.Cleanup(func() {
		newTensorEmbedding = original
	})

	sentinel := errors.New("embedding boom")
	newTensorEmbedding = func(
		*nn.TensorBackend,
		string,
		int,
		int,
		nn.Initializer,
		nn.Generator,
	) (*nn.TensorEmbeddingModule, error) {
		return nil, sentinel
	}

	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	_, err = NewModel(backend, 3, 2, nn.NewGenerator(1, 0))
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel)
	assert.ErrorContains(t, err, "create token embedding")
}

func TestNewModelWrapsRegisterError(t *testing.T) {
	original := registerModule
	t.Cleanup(func() {
		registerModule = original
	})

	sentinel := errors.New("register boom")
	registerModule = func(*nn.ModuleCollection, string, nn.Module) error {
		return sentinel
	}

	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	_, err = NewModel(backend, 3, 2, nn.NewGenerator(1, 0))
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel)
	assert.ErrorContains(t, err, "register token embedding module")
}

func TestModelNilReceiverGuards(t *testing.T) {
	var model *Model

	_, err := model.Forward([]int{0}, 1)
	require.ErrorIs(t, err, nnerr.ErrNilTensor)

	_, err = model.NewOptimizer(nil, nn.NewTensorAdamConfig())
	require.ErrorIs(t, err, nnerr.ErrNilTensor)

	assert.Nil(t, model.Parameters())

	_, err = model.StateDict()
	require.ErrorIs(t, err, nnerr.ErrNilTensor)

	err = model.LoadStateDict(nn.StateDict{})
	require.ErrorIs(t, err, nnerr.ErrNilTensor)

	err = model.WriteTrainingCheckpoint(&bytes.Buffer{}, nil, nn.ExecutionContext{})
	require.ErrorIs(t, err, nnerr.ErrNilTensor)

	_, err = model.RestoreTrainingCheckpoint(
		bytes.NewReader(nil),
		nil,
		nil,
		nn.DefaultCheckpointLimits(),
	)
	require.ErrorIs(t, err, nnerr.ErrNilTensor)

	model.Close()
}

func TestWriteTrainingCheckpointWrapsSnapshotError(t *testing.T) {
	original := newTrainingCheckpoint
	t.Cleanup(func() {
		newTrainingCheckpoint = original
	})

	sentinel := errors.New("snapshot boom")
	newTrainingCheckpoint = func(
		*nn.ModuleCollection,
		*nn.TensorAdam,
		nn.ExecutionContext,
	) (nn.TrainingCheckpoint, error) {
		return nn.TrainingCheckpoint{}, sentinel
	}

	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	model, err := NewModel(backend, 3, 2, nn.NewGenerator(1, 0))
	require.NoError(t, err)
	t.Cleanup(model.Close)

	err = model.WriteTrainingCheckpoint(&bytes.Buffer{}, nil, nn.ExecutionContext{})
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel)
	assert.ErrorContains(t, err, "snapshot training checkpoint")
}

func TestWriteTrainingCheckpointReturnsWriterError(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	model, err := NewModel(backend, 3, 2, nn.NewGenerator(1, 0))
	require.NoError(t, err)
	t.Cleanup(model.Close)

	optimizer, err := model.NewOptimizer(backend, nn.NewTensorAdamConfig())
	require.NoError(t, err)
	t.Cleanup(optimizer.Close)

	context, err := nn.NewExecutionContext(nn.Training, nn.NewGenerator(1, 0))
	require.NoError(t, err)

	err = model.WriteTrainingCheckpoint(errorWriter{}, optimizer, context)
	require.Error(t, err)
	assert.ErrorContains(t, err, "write boom")
}

func TestRestoreTrainingCheckpointWrapsReadError(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	model, err := NewModel(backend, 3, 2, nn.NewGenerator(1, 0))
	require.NoError(t, err)
	t.Cleanup(model.Close)

	_, err = model.RestoreTrainingCheckpoint(
		bytes.NewReader([]byte("not-a-checkpoint")),
		backend,
		nil,
		nn.DefaultCheckpointLimits(),
	)
	require.Error(t, err)
	assert.ErrorContains(t, err, "read training checkpoint")
}

func TestRestoreTrainingCheckpointWrapsRestoreError(t *testing.T) {
	original := restoreCheckpoint
	t.Cleanup(func() {
		restoreCheckpoint = original
	})

	sentinel := errors.New("restore boom")
	restoreCheckpoint = func(
		*nn.TensorBackend,
		*nn.ModuleCollection,
		*nn.TensorAdam,
		nn.TrainingCheckpoint,
	) (nn.ExecutionContext, error) {
		return nn.ExecutionContext{}, sentinel
	}

	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	model, err := NewModel(backend, 3, 2, nn.NewGenerator(1, 0))
	require.NoError(t, err)
	t.Cleanup(model.Close)

	optimizer, err := model.NewOptimizer(backend, nn.NewTensorAdamConfig())
	require.NoError(t, err)
	t.Cleanup(optimizer.Close)

	context, err := nn.NewExecutionContext(nn.Training, nn.NewGenerator(1, 0))
	require.NoError(t, err)

	var checkpoint bytes.Buffer
	require.NoError(t, model.WriteTrainingCheckpoint(&checkpoint, optimizer, context))

	_, err = model.RestoreTrainingCheckpoint(
		bytes.NewReader(checkpoint.Bytes()),
		backend,
		optimizer,
		nn.DefaultCheckpointLimits(),
	)
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel)
	assert.ErrorContains(t, err, "restore training checkpoint")
}

func TestCloseIsIdempotent(t *testing.T) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	require.NoError(t, err)
	t.Cleanup(backend.Close)

	model, err := NewModel(backend, 3, 2, nn.NewGenerator(1, 0))
	require.NoError(t, err)

	model.Close()
	model.Close()
}

type errorWriter struct{}

func (errorWriter) Write([]byte) (int, error) {
	return 0, errors.New("write boom")
}
