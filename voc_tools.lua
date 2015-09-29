require 'image'

local classes = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}

return {
	classes = classes,
	numClasses = #classes,

	load = function(trainval_VOCdevkit_VOC2012, test_VOCdevkit_VOC2012)
		local ffi = require 'ffi'
		local filelists = {
			train = paths.concat(trainval_VOCdevkit_VOC2012, 'ImageSets/Main/train.txt'),
			val  = paths.concat(trainval_VOCdevkit_VOC2012, 'ImageSets/Main/val.txt'),
			test = paths.concat(test_VOCdevkit_VOC2012, 'ImageSets/Main/test.txt'),
		}
		local numMaxSamples = 11000
		local mkDataset = function() return {filenames = torch.CharTensor(numMaxSamples, 16), labels = torch.FloatTensor(numMaxSamples, #classes):zero(), jpegs = {}} end
		local voc = {train = mkDataset(), val = mkDataset(), test = mkDataset()}

		for _, subset in ipairs{'train', 'val', 'test'} do
			io.input(filelists[subset])
			local count = 1
			for line in io.lines() do
				ffi.copy(voc[subset].filenames:data() + filenames:size(2) * (count - 1), line)
				  
				local f = torch.DiskFile(paths.concat(subset == 'test' and test_VOCdevkit_VOC2012 or trainval_VOCdevkit_VOC2012, 'JPEGImages', line .. '.jpg'), 'r')
				f:binary()
				f:seekEnd()
				local file_size_bytes = f:position() - 1
				f:seek(1)
				voc[subset].jpegs[count] = torch.ByteTensor(file_size_bytes)
				f:readByte(voc[subset][count].jpegs[count]:storage())
				f:close()

				count = count + 1
			end
		end   

		for _, subset in ipairs{'train', 'val'} do
		   for classInd, v in ipairs(classes) do
			  io.input(paths.concat(trainval_VOCdevkit_VOC2012, 'ImageSets/Main/'..v..'_'..subset..'.txt'))
			  local count = 1
			  for line in io.lines() do
				 if string.find(line, ' -1', 1, true) then
					voc[subset].labels[count][classInd] = -1
				 elseif string.find(line, ' 1', 1, true) then
					voc[subset].labels[count][classInd] = 1
				 end
				 count = count + 1
			  end
		   end
		end
		
		for _, subset in ipairs{'train', 'val', 'test'} do
			voc[subset].numSamples = #voc.jpegs
			voc[subset].filenames = voc[subset].filenames:narrow(1, 1, voc[subset].numSamples)
			voc[subset].labels = voc[subset].labels:narrow(1, 1, voc[subset].numSamples)
		end

		return voc
	end,

	package_submission = function(voc, subset, scores, OUT)
		assert(#(voc[subset]) == scores:size(1))

		os.execute(string.format('rm -rf %s/results', OUT))
		os.execute(string.format('mkdir -p %s/results/VOC2012/Main', OUT))
		for classLabelInd, classLabel in ipairs(classes) do
			local f = assert(io.open(string.format('%s/results/VOC2012/Main/comp2_cls_%s_%s.txt', OUT, subset, classLabel), 'w'))
			for i = 1, #(voc[subset]) do
				f:write(string.format('%s %.12f\n', voc[subset][i].filename, scores[i][classLabelInd]))
			end
			f:close()
		end
		os.execute(string.format('cd %s && tar -czf results-voc2012-%s.tar.gz results', OUT, subset))
	end,

	vis_submission = function(subset, classLabel, OUT, JPEGImages_DIR, top_k)
		top_k = top_k or 20
		local res_file_path = string.format('%s/results/VOC2012/Main/comp2_cls_%s_%s.txt', OUT, subset, classLabel)

		local scores = {}
		for line in assert(io.open(res_file_path)):lines() do
			scores[#scores + 1] = line:split(' ')
		end

		table.sort(scores, function(a, b) return -tonumber(a[2]) < -tonumber(b[2]) end)

		local top_imgs = {}
		print('K = ', top_k)
		for i = 1, top_k do
			top_imgs[i] = image.scale(image.load(paths.concat(JPEGImages_DIR, scores[i][1] .. '.jpg')), 128, 128)
			print(scores[i][2], scores[i][1])
		end

		image.display(top_imgs)
	end,

	precisionrecall = function(scores_all, labels_all)
		--adapted from VOCdevkit/VOCcode/VOCevalcls.m (VOCap.m). tested, gives equivalent results
		local function VOCap(rec, prec)
			local mrec = torch.cat(torch.cat(torch.FloatTensor({0}), rec), torch.FloatTensor({1}))
			local mpre = torch.cat(torch.cat(torch.FloatTensor({0}), prec), torch.FloatTensor({0}))
			for i=mpre:numel()-1, 1, -1 do
				mpre[i]=math.max(mpre[i], mpre[i+1])
			end

			local i = (mrec:sub(2, mrec:numel())):ne(mrec:sub(1, mrec:numel() - 1)):nonzero():squeeze(2) + 1
			local ap = (mrec:index(1, i) - mrec:index(1, i - 1)):cmul(mpre:index(1, i)):sum()

			return ap
		end

		local function VOCevalcls(out, gt)
			local so,si= (-out):sort()

			local tp=gt:index(1, si):gt(0):float()
			local fp=gt:index(1, si):lt(0):float()

			fp=fp:cumsum()
			tp=tp:cumsum()

			local rec=tp/gt:gt(0):sum()
			local prec=tp:cdiv(fp+tp)

			local ap=VOCap(rec,prec)
			return rec, prec, ap
		end

		local prec = torch.FloatTensor(scores_all:size())
		local rec = torch.FloatTensor(scores_all:size())
		local ap = torch.FloatTensor(#classes)

		for classInd = 1, #classes do
			local p, r, a = VOCevalcls(scores_all:narrow(2, classInd, 1):squeeze(), labels_all:narrow(2, classInd, 1):squeeze())
			prec:narrow(2, classInd, 1):copy(p)
			rec:narrow(2, classInd, 1):copy(r)
			ap[classInd] = a
		end

		return prec, rec, ap
	end
}
