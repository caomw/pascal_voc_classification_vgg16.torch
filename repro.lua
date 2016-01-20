require 'cunn'
require 'cudnn'
require 'loadcaffe'
require 'optim'
require 'texfuncs'

voc_tools = dofile('voc_tools.lua')
PATHS = dofile('PATHS.lua')
dofile('parallel_batch_loader.lua')
dofile('external/fbnn/fbnn/Optim.lua')
dofile('MNLLCriterion.lua')

numClasses = 20

tic = torch.tic()
print('DatasetLoader loading')
voc = voc_tools.load(PATHS.EXTERNAL.VOC_TRAINVAL_TEST)

local dataset_loader = {
	scale = {800, 608},
	bgrPixelMeans = {102.9801, 115.9465, 122.7717},
	subsets = subsets or {[true] = 'train', [false] = 'val'},
	voc = voc,

	makeBatchTable = function(self, batchSize, isTrainingPhase)
		local images = torch.FloatTensor(batchSize, 3, self.scale[2], self.scale[1])
		local labels = torch.FloatTensor(batchSize, numClasses)

		return {images, labels}
	end,

	getNumSamples = function(self, isTrainingPhase)
		return self.voc[self.subsets[isTrainingPhase]]:getNumSamples()
	end,

	loadExample = function(self, exampleIdx, isTrainingPhase)
		local	labels_loaded = self.voc[self.subsets[isTrainingPhase]]:getLabels(exampleIdx)
		local	jpeg = self.voc[self.subsets[isTrainingPhase]]:getJpegBytes(exampleIdx)
		local	scale = self.scale
		local	numRoisPerImage = self.numRoisPerImage
		local   bgr_pixel_means = self.bgrPixelMeans

		local function rescale(dhw_rgb, max_width, target_height)
			local im_scale = target_height / dhw_rgb:size(2)
			if torch.round(dhw_rgb:size(3) * im_scale) > max_width then
				im_scale = math.min(im_scale, max_width / dhw_rgb:size(3))
			end

			local scaled = image.scale(dhw_rgb, dhw_rgb:size(3) * im_scale, dhw_rgb:size(2) * im_scale):float()
			local dhw_bgr = torch.FloatTensor(scaled:size())

			dhw_bgr[1]:copy(scaled[3]):add(-bgr_pixel_means[1])
			dhw_bgr[2]:copy(scaled[2]):add(-bgr_pixel_means[2])
			dhw_bgr[3]:copy(scaled[1]):add(-bgr_pixel_means[3])

			return dhw_bgr
		end

		return function(k, batchTable)
			local images, labels = unpack(batchTable)

			labels[k]:copy(labels_loaded)

			image = image or require 'image'
			local img_decompressed = image.decompressJPG(jpeg, 3, 'byte')
			local currentSize = {width = img_decompressed:size(3), height = img_decompressed:size(2)}
			local img_processed = rescale(img_decompressed, scale[1], scale[2])
			images[k]:zero()
			images[k]:sub(1, 3, 1, img_processed:size(2), 1, img_processed:size(3)):copy(img_processed)
		end
	end
}

print('DatasetLoader loaded', torch.toc(tic))

tic = torch.tic()
print('Dataset loading')
dataset = ParallelBatchLoader(dataset_loader):setBatchSize(8)
print('Dataset loaded', torch.toc(tic))

print('Featex loading')
tic = torch.tic()
featex = nn.Sequential()
vgg16_loadcaffe = loadcaffe.load(PATHS.EXTERNAL.VGG16_PROTOTXT, PATHS.EXTERNAL.VGG16_CAFFEMODEL, 'cudnn'):float()
for i = 1, 37 do
	featex:add(vgg16_loadcaffe:get(i))
end

function convertLinear2Conv1x1(linmodule,in_size)
	local convmodule = cudnn.SpatialConvolution(linmodule.weight:size(2)/(in_size[1]*in_size[2]),linmodule.weight:size(1),in_size[1],in_size[2],1,1)
	convmodule.weight:copy(linmodule.weight)
	convmodule.bias:copy(linmodule.bias)
	return convmodule
end

featex.modules[33] = convertLinear2Conv1x1(featex.modules[33], {7, 7})
featex.modules[36] = convertLinear2Conv1x1(featex.modules[36], {1, 1})
featex:remove(32) --nn.View

print('Featex loaded', torch.toc(tic))

model = nn.Sequential()
--model:add(nn.View(4096))
--model:add(nn.Linear(4096, 2048))
model:add(cudnn.SpatialConvolution(4096,4096,1,1,1,1))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))

--model:add(nn.Linear(2048, 2048))
model:add(cudnn.SpatialConvolution(4096,4096,1,1,1,1))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))

--model:add(nn.Linear(2048, numClasses))
model:add(cudnn.SpatialConvolution(4096,numClasses,1,1,1,1))
model:add(nn.SpatialAdaptiveMaxPooling(1,1))

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

voc_tools.package_submission(PATHS.DATA, voc, 'val', 'comp2_cls', scores_all)

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

voc_tools.package_submission(PATHS.DATA, voc, 'test', 'comp2_cls', scores_all)
