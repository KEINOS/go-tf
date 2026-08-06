package tf

import (
	"testing"

	"github.com/KEINOS/go-nn/nn"
)

func BenchmarkNewModel(b *testing.B) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(backend.Close)

	b.ReportAllocs()

	for b.Loop() {
		model, modelErr := NewModel(backend, 512, 128, nn.NewGenerator(1, 0))
		if modelErr != nil {
			b.Fatal(modelErr)
		}

		model.Close()
	}
}

func BenchmarkStateDict(b *testing.B) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(backend.Close)

	model, err := NewModel(backend, 512, 128, nn.NewGenerator(2, 0))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(model.Close)

	b.ReportAllocs()

	for b.Loop() {
		_, stateErr := model.StateDict()
		if stateErr != nil {
			b.Fatal(stateErr)
		}
	}
}

func BenchmarkLoadStateDict(b *testing.B) {
	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(backend.Close)

	source, err := NewModel(backend, 512, 128, nn.NewGenerator(3, 0))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(source.Close)

	target, err := NewModel(backend, 512, 128, nn.NewGenerator(4, 0))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(target.Close)

	state, err := source.StateDict()
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()

	for b.Loop() {
		loadErr := target.LoadStateDict(state)
		if loadErr != nil {
			b.Fatal(loadErr)
		}
	}
}

func BenchmarkModelForwardSmall(b *testing.B) {
	benchmarkForward(b, []int{2, 1, 0}, 3)
}

func BenchmarkModelForwardBatch64(b *testing.B) {
	indices := make([]int, 64)

	for idx := range len(indices) {
		indices[idx] = idx % 32
	}

	benchmarkForward(b, indices, len(indices))
}

func BenchmarkModelForwardBatch256(b *testing.B) {
	indices := make([]int, 256)

	for idx := range len(indices) {
		indices[idx] = idx % 64
	}

	benchmarkForward(b, indices, len(indices))
}

func benchmarkForward(b *testing.B, indices []int, shape int) {
	b.Helper()

	backend, err := nn.NewTensorBackend(nn.UseCPU)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(backend.Close)

	model, err := NewModel(backend, 512, 128, nn.NewGenerator(5, 0))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(model.Close)

	b.ReportAllocs()

	for b.Loop() {
		node, forwardErr := model.Forward(indices, shape)
		if forwardErr != nil {
			b.Fatal(forwardErr)
		}

		_ = node.Value().Shape()
	}
}
