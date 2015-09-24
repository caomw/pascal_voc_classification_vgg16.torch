require 'cunn'
require 'cudnn'
require 'loadcaffe'
require 'optim'
require 'datamodule'
require 'texfuncs'
require 'hdf5'

voc_tools = require 'voc_tools.lua'
PATHS = require 'PATHS.lua'
require 'external/fbnn/fbnn/Optim.lua'
require 'MNLLCriterion.lua'

numClasses = 20

tic = torch.tic()
print('DatasetLoader loading')
voc = torch.load(PATHS.VOC_CACHED)
dataset_loader = {
	bgr_pixel_means = {102.9801, 115.9465, 122.7717},
	numClasses = numClasses,
	height = 224,
	width = 224,
	voc = voc,

	initOnEveryThread = function(self)
		require 'image'
	end,

	makeBatchTable = function(self, batchSize)
		local images = torch.FloatTensor(batchSize, 3, self.height, self.width)
		local labels = torch.FloatTensor(batchSize, self.numClasses)

		return {images, labels}
	end,

	getNumSamples = function(self, phase)
		return #(self.voc[phase])
	end,

	loadExample = function(self, phase, exampleIdx, batchTable, i)
		local images, labels = unpack(batchTable)
		local imgTable = self.voc[phase][exampleIdx]

		local img_decompressed = image.decompressJPG(imgTable.jpegfile, 3, 'byte')
		local img_processed = image.scale(self:preprocessImage(img_decompressed), self.width, self.height)
		images[i]:copy(img_processed)

		labels[i]:copy(imgTable.classes)

		collectgarbage()
	end,

	preprocessImage = function(self, dhw_rgb_img)
		local dhw_bgr_img = dhw_rgb_img:float():clone()
		dhw_bgr_img[{1, {}, {}}] = dhw_rgb_img[{3, {}, {}}]
		dhw_bgr_img[{3, {}, {}}] = dhw_rgb_img[{1, {}, {}}]

		for i = 1, 3 do
			dhw_bgr_img[i]:add(-self.bgr_pixel_means[i])
		end

		return dhw_bgr_img
	end,
}
print('DatasetLoader loaded', torch.toc(tic))

tic = torch.tic()
print('Dataset loading')
dataset = nn.DataModule(dataset_loader)
dataset:setBatchSize(16)
print('Dataset loaded', torch.toc(tic))

print('Featex loading')
tic = torch.tic()
vgg16_loadcaffe = loadcaffe.load(PATHS.EXTERNAL.VGG16_PROTOTXT, PATHS.EXTERNAL.VGG16_CAFFEMODEL, 'cudnn'):float()
featex = nn.Sequential()
for i = 1, 36 do --37 is ReLU
	featex:add(vgg16_loadcaffe:get(i))
end
print('Featex loaded', torch.toc(tic))

model = nn.Sequential()
model:add(nn.View(4096))
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 2048)) --model:add(cudnn.SpatialConvolution(4096,4096,1,1,1,1))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))

model:add(nn.Linear(2048, 2048)) --model:add(cudnn.SpatialConvolution(4096,4096,1,1,1,1))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))

model:add(nn.Linear(2048, numClasses)) --model:add(cudnn.SpatialConvolution(4096,numClasses,1,1,1,1))

featex:cuda()
jittering = nn.TexFunCropFlip(0):cuda()
model:cuda()
criterion = MultiClassNLLCriterion():cuda()

optimState = {learningRate = 0.001, momentum = 0.9, weightDecay = 5e-4}
model:apply(function (x) x.for_each = x.apply end)
optimizer = nn.Optim(model, optimState)
optimalg = optim.adagrad
logger = optim.Logger(PATHS.TRAINING_CURVE)

lastValidationCost = 0
APs, mAPs, validationCosts = {}, {}, {}
gpuInput, gpuLabels = torch.CudaTensor(), torch.CudaTensor()

for epoch = 1, 10 do
	if epoch > 5 then
		optimState.learningRate = 0.0005
		optimizer:setParameters(optimState)
	end

	tic = torch.tic()	
	dataset:training()
	model:training()
	for batchIdx = 1, dataset:getNumBatches() do
		batch, labels = unpack(dataset:forward())
		gpuInput:resize(batch:size()):copy(batch)
		gpuLabels:resize(labels:size()):copy(labels)
		
		jittered = jittering:forward(gpuInput)
		features = featex:forward(jittered)
		cost = optimizer:optimize(optimalg, features, gpuLabels, criterion)

		logger:add{['cost'] = cost, ['validation cost'] = lastValidationCost, ['epoch'] = epoch, ['batch'] = batchIdx}
		print('epoch', epoch, 'batch', batchIdx, cost)
	end
	print('train', torch.toc(tic))

	dataset:evaluate()
	model:evaluate()
	lastValidationCost = 0
	scores_all = torch.FloatTensor(dataset:getNumSamples(), numClasses)
	labels_all = torch.FloatTensor(scores_all:size())
	for batchIdx = 1, dataset:getNumBatches() do
		batch, labels = unpack(dataset:forward())
		gpuInput:resize(batch:size()):copy(batch)
		gpuLabels:resize(labels:size()):copy(labels)
		
		features = featex:forward(gpuInput)
		scores = model:forward(features)
		cost = criterion:forward(scores, gpuLabels)

		lastValidationCost = lastValidationCost + cost
		scores_all:narrow(1, 1 + (batchIdx - 1) * dataset:getBatchSize(), batch:size(1)):copy(scores)
		labels_all:narrow(1, 1 + (batchIdx - 1) * dataset:getBatchSize(), batch:size(1)):copy(labels)

		print('val', 'epoch', epoch, 'batch', batchIdx, cost)
	end

	lastValidationCost = lastValidationCost / dataset:getNumBatches()
	
	_, _, perClassAp = voc_tools.precisionrecall(scores_all, labels_all)
	print('per class AP :', perClassAp)
	print('mean AP :', perClassAp:mean())
	
	APs[#APs + 1] = perClassAp
	mAPs[#mAPs + 1] = perClassAp:mean()
	validationCosts[#validationCosts + 1] = lastValidationCost
	
	torch.save(PATHS.LOG, {logger = logger, APs = APs, mAPs = mAPs, validationCosts = validationCosts})
	torch.save(PATHS.MODEL, model)
end

voc_tools.package_submission(voc, 'val', scores_all, PATHS.DATA)

dataset:testing()
model:evaluate()
scores_all = torch.FloatTensor(dataset:getNumSamples(), numClasses)
for batchIdx = 1, dataset:getNumBatches() do
	batch, labels = unpack(dataset:forward())
	gpuInput:resize(batch:size()):copy(batch)
	gpuLabels:resize(labels:size()):copy(labels)
	
	features = featex:forward(gpuInput)
	scores = model:forward(features)

	scores_all:narrow(1, 1 + (batchIdx - 1) * dataset:getBatchSize(), batch:size(1)):copy(scores)
	print('test', 'batch', batchIdx)
end

voc_tools.package_submission(voc, 'test', scores_all, PATHS.DATA)
