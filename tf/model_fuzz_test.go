package tf

import (
	"errors"
	"testing"

	"github.com/KEINOS/go-nn/nn"
	"github.com/KEINOS/go-nn/nn/nnerr"
)

func FuzzNewModelDimensions(f *testing.F) {
	f.Add(0, 2)
	f.Add(2, 0)
	f.Add(8, 4)
	f.Add(-1, 3)

	f.Fuzz(func(t *testing.T, rawVocab, rawEmbedding int) {
		vocabSize := boundedDimension(rawVocab, 512)
		embeddingSize := boundedDimension(rawEmbedding, 128)

		backend, err := nn.NewTensorBackend(nn.UseCPU)
		if err != nil {
			t.Fatalf("create backend: %v", err)
		}
		t.Cleanup(backend.Close)

		model, err := NewModel(backend, vocabSize, embeddingSize, nn.NewGenerator(10, 0))

		if vocabSize <= 0 || embeddingSize <= 0 {
			if !errors.Is(err, nnerr.ErrInvalidDimension) {
				t.Fatalf(
					"want invalid dimension error for vocab=%d embedding=%d, got: %v",
					vocabSize,
					embeddingSize,
					err,
				)
			}

			return
		}

		if err != nil {
			t.Fatalf(
				"unexpected NewModel error for vocab=%d embedding=%d: %v",
				vocabSize,
				embeddingSize,
				err,
			)
		}

		model.Close()
	})
}

func FuzzModelForwardIndices(f *testing.F) {
	f.Add(8, 4, 0, 1, 2, 3)
	f.Add(8, 4, -1, 1, 2, 3)
	f.Add(8, 4, 9, 10, 11, 3)
	f.Add(8, 4, 0, 1, 2, 0)

	f.Fuzz(func(t *testing.T, rawVocab, rawEmbedding, rawA, rawB, rawC, rawShape int) {
		vocabSize := positiveDimension(rawVocab, 64)
		embeddingSize := positiveDimension(rawEmbedding, 16)

		backend, err := nn.NewTensorBackend(nn.UseCPU)
		if err != nil {
			t.Fatalf("create backend: %v", err)
		}
		t.Cleanup(backend.Close)

		model, err := NewModel(backend, vocabSize, embeddingSize, nn.NewGenerator(11, 0))
		if err != nil {
			t.Fatalf("create model: %v", err)
		}
		t.Cleanup(model.Close)

		indices := []int{
			fuzzIndex(rawA, vocabSize),
			fuzzIndex(rawB, vocabSize),
			fuzzIndex(rawC, vocabSize),
		}

		shape := positiveDimension(rawShape, len(indices)+2)

		defer func() {
			recovered := recover()
			if recovered != nil {
				t.Fatalf(
					"Forward panicked for vocab=%d embedding=%d indices=%v shape=%d: %v",
					vocabSize,
					embeddingSize,
					indices,
					shape,
					recovered,
				)
			}
		}()

		node, forwardErr := model.Forward(indices, shape)
		if forwardErr != nil {
			return
		}

		if node == nil {
			t.Fatalf("Forward succeeded with nil node")
		}

		gotShape := node.Value().Shape()
		if len(gotShape) == 0 {
			t.Fatalf("Forward returned empty output shape")
		}

		lastDim := gotShape[len(gotShape)-1]
		if lastDim != embeddingSize {
			t.Fatalf(
				"unexpected last dimension: got %d want %d (shape=%v)",
				lastDim,
				embeddingSize,
				gotShape,
			)
		}
	})
}

func boundedDimension(raw, max int) int {
	if max <= 0 {
		return 1
	}

	return (raw % (max + 8)) - 4
}

func positiveDimension(raw, max int) int {
	if max <= 0 {
		return 1
	}

	value := raw % max
	if value < 0 {
		value = -value
	}

	return value + 1
}

func fuzzIndex(raw, vocabSize int) int {
	if vocabSize <= 0 {
		return 0
	}

	pattern := raw % 4
	if pattern < 0 {
		pattern = -pattern
	}

	switch pattern {
	case 0:
		return raw % vocabSize
	case 1:
		return -(raw%vocabSize + 1)
	case 2:
		return vocabSize + raw%vocabSize
	default:
		return raw
	}
}
