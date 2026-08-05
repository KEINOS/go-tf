// Package tf composes model-specific Transformer components from go-nn.
package tf

import (
	"fmt"
	"io"

	"github.com/KEINOS/go-nn/nn"
	"github.com/KEINOS/go-nn/nn/nnerr"
)

// Model is the initial go-tf consumer of go-nn Module and Parameter state.
type Model struct {
	modules   *nn.ModuleCollection
	embedding *nn.TensorEmbeddingModule
}

// NewModel creates a minimal token-embedding model state on backend.
func NewModel(
	backend *nn.TensorBackend,
	vocabularySize, embeddingSize int,
	generator nn.Generator,
) (*Model, error) {
	if vocabularySize <= 0 || embeddingSize <= 0 {
		return nil, nnerr.ErrInvalidDimension
	}

	modules, err := nn.NewModuleCollection(backend)
	if err != nil {
		return nil, fmt.Errorf("create model modules: %w", err)
	}

	embedding, err := nn.NewTensorEmbeddingModule(
		backend,
		"token.embedding",
		vocabularySize,
		embeddingSize,
		nn.UniformInitializer(-0.02, 0.02),
		generator,
	)
	if err != nil {
		modules.Close()

		return nil, fmt.Errorf("create token embedding: %w", err)
	}

	err = modules.RegisterModule("token.embedding", embedding)
	if err != nil {
		embedding.Close()
		modules.Close()

		return nil, fmt.Errorf("register token embedding module: %w", err)
	}

	return &Model{modules: modules, embedding: embedding}, nil
}

// Forward gathers token embedding rows with an explicit index shape.
func (model *Model) Forward(indices []int, indexShape ...int) (*nn.TensorNode, error) {
	if model == nil || model.embedding == nil {
		return nil, nnerr.ErrNilTensor
	}

	return model.embedding.Forward(indices, indexShape...)
}

// NewOptimizer freezes the Model's current Parameter membership.
func (model *Model) NewOptimizer(
	backend *nn.TensorBackend,
	config nn.TensorAdamConfig,
) (*nn.TensorAdam, error) {
	if model == nil || model.modules == nil {
		return nil, nnerr.ErrNilTensor
	}

	return nn.NewTensorAdam(backend, model.modules, config)
}

// Parameters returns canonical model Parameters in stable name order.
func (model *Model) Parameters() []*nn.Parameter {
	if model == nil || model.modules == nil {
		return nil
	}

	return model.modules.Parameters()
}

// StateDict returns a device-independent immutable model snapshot.
func (model *Model) StateDict() (nn.StateDict, error) {
	if model == nil || model.modules == nil {
		return nn.StateDict{}, nnerr.ErrNilTensor
	}

	return model.modules.StateDict()
}

// LoadStateDict strictly and atomically restores model Parameters.
func (model *Model) LoadStateDict(state nn.StateDict) error {
	if model == nil || model.modules == nil {
		return nnerr.ErrNilTensor
	}

	return model.modules.LoadStateDict(state)
}

// WriteTrainingCheckpoint snapshots and writes complete reproducible state.
func (model *Model) WriteTrainingCheckpoint(
	writer io.Writer,
	optimizer *nn.TensorAdam,
	context nn.ExecutionContext,
) error {
	if model == nil || model.modules == nil {
		return nnerr.ErrNilTensor
	}

	checkpoint, err := nn.NewTrainingCheckpoint(model.modules, optimizer, context)
	if err != nil {
		return fmt.Errorf("snapshot training checkpoint: %w", err)
	}

	return nn.WriteTrainingCheckpoint(writer, checkpoint)
}

// RestoreTrainingCheckpoint atomically restores model and optimizer state.
func (model *Model) RestoreTrainingCheckpoint(
	reader io.Reader,
	backend *nn.TensorBackend,
	optimizer *nn.TensorAdam,
	limits nn.CheckpointLimits,
) (nn.ExecutionContext, error) {
	if model == nil || model.modules == nil {
		return nn.ExecutionContext{}, nnerr.ErrNilTensor
	}

	checkpoint, err := nn.ReadTrainingCheckpoint(reader, limits)
	if err != nil {
		return nn.ExecutionContext{}, fmt.Errorf("read training checkpoint: %w", err)
	}

	context, err := nn.RestoreTrainingCheckpoint(backend, model.modules, optimizer, checkpoint)
	if err != nil {
		return nn.ExecutionContext{}, fmt.Errorf("restore training checkpoint: %w", err)
	}

	return context, nil
}

// Close releases model-owned Parameter state but not its TensorBackend.
func (model *Model) Close() {
	if model == nil || model.modules == nil {
		return
	}

	model.modules.Close()
}

var _ nn.Module = (*Model)(nil)
