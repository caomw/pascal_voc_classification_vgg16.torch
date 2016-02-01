require 'nn'

local ParallelBatchLoader, parent = torch.class('ParallelBatchLoader', 'nn.Module')

function ParallelBatchLoader:__init(loader, nThreads)
	parent.__init(self)

	self.loader = loader
	self.nThreads = nThreads or 16

	self.nextBatchIdx = 1
	self.preloadedBatchIdx = nil
	
	self.batchSize = nil
	self.batchBuffers = nil
	self.currentBufferIdx = 1
	
	local threads = require 'threads'
	threads.Threads.serialization('threads.sharedserialize')
	self.jobQueue = threads.Threads(self.nThreads)

	self:evaluate()
end

function ParallelBatchLoader:loadBatch(exampleIdxBegin)
	self.jobQueue:synchronize()

	self.currentBufferIdx = 3 - self.currentBufferIdx
	local batchTable = self.batchBuffers[self.train][self.currentBufferIdx]
	local isTrainingPhase = self.train

	for exampleIndexInBatch = 1, self.batchSize do
		local exampleIdx = isTrainingPhase and torch.random(1, self:getNumSamples()) or (exampleIdxBegin - 1 + exampleIndexInBatch)
		local fillBatchTable = self.loader:loadExample(exampleIdx, isTrainingPhase)
		self.jobQueue:addjob(function()	fillBatchTable(exampleIndexInBatch, batchTable) end)
	end
end

function ParallelBatchLoader:getBatch(batchIdx)
	batchIdx = batchIdx or 1
	assert(batchIdx <= self:getNumBatches())
	
	local exampleIdxBegin = 1 + (batchIdx - 1) * self.batchSize
	local exampleIdxEnd = 1 + math.min(batchIdx * self.batchSize, self:getNumSamples())
	local effectiveBatchSize = exampleIdxEnd - exampleIdxBegin
	local oldBatchSize = self.batchSize

	if batchIdx ~= self.preloadedBatchIdx[self.train] or effectiveBatchSize ~= self.batchSize then
		self:setBatchSize(effectiveBatchSize)
		self.preloadedBatchIdx[self.train] = batchIdx
		self:loadBatch(exampleIdxBegin)
	end

	self.jobQueue:synchronize()
	local loadedBatchTable = self.batchBuffers[self.train][self.currentBufferIdx]

	if self.batchSize ~= oldBatchSize then
		self:setBatchSize(oldBatchSize)
	end

	local nextBatchIdx = batchIdx + 1
	if nextBatchIdx < self:getNumBatches() then
		self.preloadedBatchIdx[self.train] = nextBatchIdx
		self:loadBatch(exampleIdxBegin + self.batchSize)
	end

	return loadedBatchTable
end

function ParallelBatchLoader:updateOutput()
	assert(self.batchSize)
	local batchIdx = self.nextBatchIdx[self.train]
	self.output = self:getBatch(batchIdx)
	self.nextBatchIdx[self.train] = batchIdx + 1
	return self.output
end

function ParallelBatchLoader:setBatchSize(batchSize)
	self.batchSize = batchSize
	self.batchBuffers = {[true] = {self.loader:makeBatchTable(batchSize, true), self.loader:makeBatchTable(batchSize, true)}, [false] = {self.loader:makeBatchTable(batchSize, false), self.loader:makeBatchTable(batchSize, false)}}
	return self
end

function ParallelBatchLoader:getBatchSize()
	return self.batchSize
end

function ParallelBatchLoader:getNumBatches()
	return torch.ceil(self:getNumSamples() / assert(self.batchSize))
end

function ParallelBatchLoader:getNumSamples()
	return self.loader:getNumSamples(self.train)
end

function ParallelBatchLoader:training()
	parent:training()
	self.nextBatchIdx[self.train] = 1
end

function ParallelBatchLoader:evaluate()
	parent:evaluate()
	self.nextBatchIdx[self.train] = 1
end
