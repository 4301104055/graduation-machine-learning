var sample_training_instance = function() {
  // find an unloaded batch
  var bi = Math.floor(Math.random()*loaded_train_batches.length);
  var b = loaded_train_batches[bi];
  var k = Math.floor(Math.random()*num_samples_per_batch); // sample within the batch
  var n = b*num_samples_per_batch+k;

  // load more batches over time
  if(step_num%(2 * num_samples_per_batch)===0 && step_num>0) {
    for(var i=0;i<num_batches;i++) {
      if(!loaded[i]) {
        // load it
        load_data_batch(i);
        break; // okay for now
      }
    }
  }
  // fetch the appropriate row of the training image and reshape into a Vol
  var p = img_data[b].data;
  var x = new convnetjs.Vol(image_dimension,image_dimension,image_channels,0.0);
  var W = image_dimension*image_dimension;
  var j=0;
  for(var dc=0;dc<image_channels;dc++) {
    var i=0;
    for(var xc=0;xc<image_dimension;xc++) {
      for(var yc=0;yc<image_dimension;yc++) {
        var ix = ((W * k) + i) * 4 + dc;
        x.set(yc,xc,dc,p[ix]/255.0-0.5);
        i++;
      }
    }
  }

  if(random_position){
    var dx = Math.floor(Math.random()*5-2);
    var dy = Math.floor(Math.random()*5-2);
    x = convnetjs.augment(x, image_dimension, dx, dy, false); //maybe change position
  }

  if(random_flip){
    x = convnetjs.augment(x, image_dimension, 0, 0, Math.random()<0.5); //maybe flip horizontally
  }

  var isval = use_validation_data && n%10===0 ? true : false;
  return {x:x, label:labels[n], isval:isval};
}
// sample a random testing instance
var sample_test_instance = function() {

  var b = test_batch;
  var k = Math.floor(Math.random()*num_image_test_set);
  var n = b*num_samples_per_batch+k;

  var p = img_data[b].data;
  var x = new convnetjs.Vol(image_dimension,image_dimension,image_channels,0.0);
  var W = image_dimension*image_dimension;
  var j=0;
  for(var dc=0;dc<image_channels;dc++) {
    var i=0;
    for(var xc=0;xc<image_dimension;xc++) {
      for(var yc=0;yc<image_dimension;yc++) {
        var ix = ((W * k) + i) * 4 + dc;
        x.set(yc,xc,dc,p[ix]/255.0-0.5);
        i++;
      }
    }
  }
console.log("------------"+n+"---"+labels[n]);
  // distort position and maybe flip
  var xs = [];
  
  if (random_flip || random_position){
    for(var k=0;k<6;k++) {
      var test_variation = x;
      if(random_position){
        var dx = Math.floor(Math.random()*5-2);
        var dy = Math.floor(Math.random()*5-2);
        test_variation = convnetjs.augment(test_variation, image_dimension, dx, dy, false);
      }
      
      if(random_flip){
        test_variation = convnetjs.augment(test_variation, image_dimension, 0, 0, Math.random()<0.5); 
      }

      xs.push(test_variation);
    }
  }else{
    xs.push(x, image_dimension, 0, 0, false); // push an un-augmented copy
  }
  
  // return multiple augmentations, and we will average the network over them
  // to increase performance
  return {x:xs, label:labels[n]};
}

var data_img_elts = new Array(num_batches);
var img_data = new Array(num_batches);
var loaded = new Array(num_batches);
var loaded_train_batches = [];
// int main
$(window).load(function() {
  $("#newnet").val(t);
  eval($("#newnet").val());
  update_net_param_display();
});

var start_fun = function() {
  if(loaded[0] && loaded[test_batch]) { 
    console.log('starting!'); 
    setInterval(load_and_step, 0); // lets go!
  }
  else { setTimeout(start_fun, 1000); } // keep checking
}

var load_data_batch = function(batch_num) {
  // Load the dataset with JS in background
  data_img_elts[batch_num] = new Image();
  var data_img_elt = data_img_elts[batch_num];

  data_img_elt.onload = function() { 
    var data_canvas = document.createElement('canvas');
    data_canvas.width = data_img_elt.width;
    data_canvas.height = data_img_elt.height;
    console.log(data_canvas.width + "---" + data_canvas.height);
    var data_ctx = data_canvas.getContext("2d");
    data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
    img_data[batch_num] = data_ctx.getImageData(0, 0, data_img_elt.width, data_img_elt.height);
    loaded[batch_num] = true;
    if(batch_num < test_batch) { loaded_train_batches.push(batch_num); }
    console.log('finished loading data batch ' + batch_num);
  };
  data_img_elt.src = "../"+ dataset_name + "/" + dataset_name + "_batch_" + batch_num + ".png";
}

var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;

// elt is the element to add all the canvas activation drawings into
// A is the Vol() to use
// scale is a multiplier to make the visualizations larger. Make higher for larger pictures
// if grads is true then gradients are used instead
var draw_activations = function(elt, A, scale, grads) {

  var s = scale || 2; // scale
      var draw_grads = false;
      if(typeof(grads) !== 'undefined') draw_grads = grads;

      // get max and min activation to scale the maps automatically
      var w = draw_grads ? A.dw : A.w;
      var mm = maxmin(w);

      // create the canvas elements, draw and add to DOM
      for(var d=0;d<A.depth;d++) {

        var canv = document.createElement('canvas');
        canv.className = 'actmap';
        var W = A.sx * s;
        var H = A.sy * s;
        canv.width = W;
        canv.height = H;
        var ctx = canv.getContext('2d');
        var g = ctx.createImageData(W, H);

        for(var x=0;x<A.sx;x++) {
          for(var y=0;y<A.sy;y++) {
            if(draw_grads) {
              var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
            } else {
              var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);  
            }
            for(var dx=0;dx<s;dx++) {
              for(var dy=0;dy<s;dy++) {
                var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
                for(var i=0;i<3;i++) { g.data[pp + i] = dval; } // rgb
                g.data[pp+3] = 255; // alpha channel
              }
            }
          }
        }
        ctx.putImageData(g, 0, 0);
        elt.appendChild(canv);
      } 
}
var draw_activations_COLOR = function(elt, A, scale, grads) {
    var s = scale || 2; // scale
    var draw_grads = false;
    if(typeof(grads) !== 'undefined') draw_grads = grads;

    // get max and min activation to scale the maps automatically
    var w = draw_grads ? A.dw : A.w;
    var mm = maxmin(w);

    var canv = document.createElement('canvas');
    canv.className = 'actmap';
    var W = A.sx * s;
    var H = A.sy * s;
    canv.width = W;
    canv.height = H;
    var ctx = canv.getContext('2d');
    var g = ctx.createImageData(W, H);
    for(var d=0;d<3;d++) {
      for(var x=0;x<A.sx;x++) {
        for(var y=0;y<A.sy;y++) {
          if(draw_grads) {
            var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
          } else {
            var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);  
          }
          for(var dx=0;dx<s;dx++) {
            for(var dy=0;dy<s;dy++) {
              var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
              g.data[pp + d] = dval;
              if(d===0) g.data[pp+3] = 255; // alpha channel
            }
          }
        }
      }
    }
    ctx.putImageData(g, 0, 0);
    elt.appendChild(canv);
}
var visualize_activations = function(net, elt) {

  // clear the element
  elt.innerHTML = "";

  // show activations in each layer
  var N = net.layers.length;
  for(var i=0;i<N;i++) {
    var L = net.layers[i];

    var layer_div = document.createElement('div');

    // visualize activations
    var activations_div = document.createElement('div');
    activations_div.appendChild(document.createTextNode('Activations:'));
    activations_div.appendChild(document.createElement('br'));
    activations_div.className = 'layer_act';
    var scale = 0; //ch???nh th??ng s??? hi???n th??? h??nh ???nh trong canvas

    if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax

    if(image_dimension <= 32)
      scale = 2;
    else
      scale = 1;
    // HACK to draw in color in input layer
    if(i===0) {
      draw_activations_COLOR(activations_div, L.out_act, scale);
      draw_activations_COLOR(activations_div, L.out_act, scale, true);

      
      // visualize positive and negative components of the gradient separately
      var dd = L.out_act.clone();
      var ni = L.out_act.w.length;
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq > 0 ? dwq : 0.0; }
      draw_activations_COLOR(activations_div, dd, scale);
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq < 0 ? -dwq : 0.0; }
      draw_activations_COLOR(activations_div, dd, scale);
      

      
      // visualize what the network would like the image to look like more
      var dd = L.out_act.clone();
      var ni = L.out_act.w.length;
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] -= 20*dwq; }
      draw_activations_COLOR(activations_div, dd, scale);
      

      
      // visualize gradient magnitude
      var dd = L.out_act.clone();
      var ni = L.out_act.w.length;
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq*dwq; }
      draw_activations_COLOR(activations_div, dd, scale);
      

    } else {
      draw_activations(activations_div, L.out_act, scale);
    } 

    // visualize data gradients
    if(L.layer_type !== 'softmax' && L.layer_type !== 'input' ) {
      var grad_div = document.createElement('div');
      grad_div.appendChild(document.createTextNode('Activation Gradients:'));
      grad_div.appendChild(document.createElement('br'));
      grad_div.className = 'layer_grad';
      var scale = 1;
      if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax

      draw_activations(grad_div, L.out_act, scale, true);
      activations_div.appendChild(grad_div);
    }

    // visualize filters if they are of reasonable size
    if(L.layer_type === 'conv') {
      var filters_div = document.createElement('div');
      if(L.filters[0].sx>3) {
        // actual weights
        filters_div.appendChild(document.createTextNode('Weights:'));
        filters_div.appendChild(document.createElement('br'));
        for(var j=0;j<L.filters.length;j++) {
          // HACK to draw in color for first layer conv filters
          if(i===1) {
            draw_activations_COLOR(filters_div, L.filters[j], 4);
          } else {
            filters_div.appendChild(document.createTextNode('('));
            draw_activations(filters_div, L.filters[j], 4);
            filters_div.appendChild(document.createTextNode(')'));
          }
        }
        // gradients
        filters_div.appendChild(document.createElement('br'));
        filters_div.appendChild(document.createTextNode('Weight Gradients:'));
        filters_div.appendChild(document.createElement('br'));
        for(var j=0;j<L.filters.length;j++) {
          if(i===1) { draw_activations_COLOR(filters_div, L.filters[j], 4, true); }
          else {
            filters_div.appendChild(document.createTextNode('('));
            draw_activations(filters_div, L.filters[j], 4, true);
            filters_div.appendChild(document.createTextNode(')'));
          }
        }
      } else {
        filters_div.appendChild(document.createTextNode('Weights hidden, too small'));
      }
      activations_div.appendChild(filters_div);
    }
    layer_div.appendChild(activations_div);

    // print some stats on left of the layer
    layer_div.className = 'layer ' + 'lt' + L.layer_type;
    var title_div = document.createElement('div');
    title_div.className = 'ltitle'
    var t = L.layer_type + ' (' + L.out_sx + 'x' + L.out_sy + 'x' + L.out_depth + ')';
    title_div.appendChild(document.createTextNode(t));
    layer_div.appendChild(title_div);

    if(L.layer_type==='conv') {
      var t = 'filter size ' + L.filters[0].sx + 'x' + L.filters[0].sy + 'x' + L.filters[0].depth + ', stride ' + L.stride;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }
    if(L.layer_type==='pool') {
      var t = 'pooling size ' + L.sx + 'x' + L.sy + ', stride ' + L.stride;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }

    // find min, max activations and display them
    var mma = maxmin(L.out_act.w);
    var t = 'max activation: ' + f2t(mma.maxv) + ', min: ' + f2t(mma.minv);
    layer_div.appendChild(document.createTextNode(t));
    layer_div.appendChild(document.createElement('br'));

    var mma = maxmin(L.out_act.dw);
    var t = 'max gradient: ' + f2t(mma.maxv) + ', min: ' + f2t(mma.minv);
    layer_div.appendChild(document.createTextNode(t));
    layer_div.appendChild(document.createElement('br'));

    // number of parameters
    if(L.layer_type==='conv' || L.layer_type==='local') {
      var tot_params = L.sx*L.sy*L.in_depth*L.filters.length + L.filters.length;
      var t = 'parameters: ' + L.filters.length + 'x' + L.sx + 'x' + L.sy + 'x' + L.in_depth + '+' + L.filters.length + ' = ' + tot_params;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }
    if(L.layer_type==='fc') {
      var tot_params = L.num_inputs*L.filters.length + L.filters.length;
      var t = 'parameters: ' + L.filters.length + 'x' + L.num_inputs + '+' + L.filters.length + ' = ' + tot_params;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }

    // css madness needed here...
    var clear = document.createElement('div');
    clear.className = 'clear';
    layer_div.appendChild(clear);

    elt.appendChild(layer_div);
  }
}

// loads a training image and trains on it with the network
var pause = false;
var load_and_step = function() {
  if(pause) return; 
  var sample = sample_training_instance();
  step(sample); // process this image 
  setTimeout(load_and_step, 1000); // schedule the next iteration
}
// evaluate current network on test set
var max_acc = parseFloat(predic_maxacc).toFixed(4);
var test_predict = function() {
  $("#testset_acc").empty();
  var num_classes = net.layers[net.layers.length-1].out_depth;
  var num_total = 0;
  var num_correct = 0;

  // l???y h??nh ???nh ki???m tra ng???u nhi??n
  for(num=0;num<10;num++) {
    var sample = sample_test_instance();
    var y = sample.label;  // ground truth label

    // forward prop it through the network
    var aavg = new convnetjs.Vol(1,1,num_classes,0.0);
    // ensures we always have a list, regardless if above returns single item or list
    var xs = [].concat(sample.x);
    var n = xs.length;
    for(var i=0;i<n;i++) {
      var a = net.forward(xs[i]);
      aavg.addFrom(a);
    }
    var preds = [];
    for(var k=0;k<aavg.w.length;k++) { preds.push({k:k,p:aavg.w[k]}); }
    preds.sort(function(a,b){return a.p<b.p ? 1:-1;});
  
    var correct = preds[0].k===y;
      //console.log(preds[0].k +"--------"+ y);
    if(correct){
      num_correct++;
    }
    num_total++;
      
    var div = document.createElement('div');
    div.className = 'testdiv';

    // draw the image into a canvas
    var scale = 0;
    if(image_dimension <= 32)
      scale = 2;
    else
      scale = 1;
    draw_activations_COLOR(div, xs[0], scale); // draw Vol into canv

    // add predictions
    var probsdiv = document.createElement('div');
    div.className = 'probsdiv';
    //div.style ='display: inline-block; overflow: hidden; width: 150;';
    var pad_top = image_dimension/2 - image_dimension/4;
    var t ='<div style = "padding-top:'+ pad_top +'px; float:left;">';
    for(var k=0;k<3;k++) {
      var col = preds[k].k===y ? 'rgb(85,187,85)' : 'rgb(187,85,85)';
      t += '<div class=\"pp\" style=\"width:' + parseFloat(preds[k].p/n*100).toFixed(2) + 'px; margin-left: 10px; background-color:' + col + ';\">' + classes_txt[preds[k].k] + '</div>'
    }
    t+= '</div>';
    probsdiv.innerHTML = t;
    div.appendChild(probsdiv);
    // add it into DOM
    $(div).prependTo($("#testset_vis")).hide().fadeIn(100).slideDown(100);
    if($(".probsdiv").length>10) {
      $("#testset_vis > .probsdiv").last().remove(); // pop to keep upper bound of shown items
    }
  }  
  testAccWindow.add(num_correct/num_total);
  var testAccvalue = parseFloat(testAccWindow.get_average()*100).toFixed(4);
  if ($('#isSaveInfo').prop('checked')){
    if(step_num!=0 && testAccvalue > max_acc){
      save_network_to_jsonfile(dataset_name);
      save_information_when_get_maxacc(start_time);
      save_infor_at_step(arrStepInfo);

      max_acc = testAccvalue;
      console.log(end_time);
    }
  }
  console.log(start_time);
  $("#testset_acc").text('Test accuracy based on last ' +  num_image_test_set + ' test images: ' + testAccvalue + "%");
}
// start time + 60
var settingTime = function(){
    
}
//-----------------------------------------bi???u ?????--------------------------------------------

//X??a bi???u ?????
var clear_graph = function() {
  lossGraph = new cnnvis.Graph();
  trainGraph = new cnnvis.Graph();
  testGraph = new cnnvis.Graph();
}

var lossGraph = new cnnvis.Graph();
var xLossWindow = new cnnutil.Window(100);
var wLossWindow = new cnnutil.Window(100);
// New
var trainGraphLabel = ['Train_Acc', 'validation_acc'];
var trainGraph = new cnnvis.MultiGraph(trainGraphLabel);
var trainAccWindow = new cnnutil.Window(100);
var wtrainAccWindow = new cnnutil.Window(100);

var valAccWindow = new cnnutil.Window(100);
// New
var testGraph = new cnnvis.Graph();
var testAccWindow = new cnnutil.Window(50, 1);
var wtestAccWindow = new cnnutil.Window(50, 1);


var start_forward_time = 0;
var backprop_time = 0;
var step_num = 0;
var arrStepInfo = [];
var step = function(sample) {
  var x = sample.x;
  var y = sample.label;

  if(sample.isval) {
    // use x to build our estimate of validation error
    net.forward(x);
    var yhat = net.getPrediction();
    var val_acc = yhat === y ? 1.0 : 0.0;
    valAccWindow.add(val_acc);
    return; // get out
  }

  // train on it with network
  var stats = trainer.train(x, y);
  var lossx = stats.cost_loss;
  var lossw = stats.l2_decay_loss;
  start_forward_time = stats.fwd_time;
  backprop_time = stats.bwd_time;
  // keep track of stats such as the average training error and loss
  var yhat = net.getPrediction();
  var train_acc = yhat === y ? 1.0 : 0.0;
  xLossWindow.add(lossx);
  wLossWindow.add(lossw);
  trainAccWindow.add(train_acc);

  // visualize training status
  var train_elt = document.getElementById("trainstats");
  train_elt.innerHTML = '';
  var t = 'Forward time per example: ' + start_forward_time + 'ms';
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Backprop time per example: ' + backprop_time + 'ms';
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Classification loss: ' + f2t(xLossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'L2 Weight decay loss: ' + f2t(wLossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Training accuracy: ' + f2t(trainAccWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Validation accuracy: ' + f2t(valAccWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Examples seen: ' + step_num;
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));

  var trainGraphDataset = [];
  trainGraphDataset.push(trainAccWindow.get_average());
  trainGraphDataset.push(valAccWindow.get_average());
  // visualize activations
  if(step_num % 100 === 0) {
    var vis_elt = document.getElementById("visnet");
    visualize_activations(net, vis_elt);
  }
  // log progress to graph, (full loss)
  if(step_num % 200 === 0) {
    var xa = xLossWindow.get_average();
    var xw = wLossWindow.get_average();
    if(xa >= 0 && xw >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
        lossGraph.add(step_num, xa + xw);
        lossGraph.drawSelf(document.getElementById("lossgraph"));
    }
    console.log(trainGraphDataset.length);
    console.log(trainGraphDataset);
    trainGraph.add(step_num, trainGraphDataset);
    trainGraph.drawSelf(document.getElementById("training-acc-chart"));
  }

  // run prediction on test set
  if((step_num % 100 === 0 && step_num > 0) || step_num===100) {
    test_predict();
    testGraph.add(step_num, testAccWindow.get_average());
    testGraph.drawSelf(document.getElementById("Testing-acc-chart"));
    var inforObject = infor_at_step(step_num, trainAccWindow.get_average(), valAccWindow.get_average(), testAccWindow.get_average(), xLossWindow.get_average() + wLossWindow.get_average());
    arrStepInfo.push(inforObject);
  }
  step_num++;
}
// user settings 
var change_lr = function() {
  trainer.learning_rate = parseFloat(document.getElementById("lr_input").value);
  update_net_param_display();
}
var change_momentum = function() {
  trainer.momentum = parseFloat(document.getElementById("momentum_input").value);
  update_net_param_display();
}
var change_batch_size = function() {
  trainer.batch_size = parseFloat(document.getElementById("batch_size_input").value);
  update_net_param_display();
}
var change_decay = function() {
  trainer.l2_decay = parseFloat(document.getElementById("decay_input").value);
  update_net_param_display();
}
var update_net_param_display = function() {
  document.getElementById('lr_input').value = trainer.learning_rate;
  document.getElementById('momentum_input').value = trainer.momentum;
  document.getElementById('batch_size_input').value = trainer.batch_size;
  document.getElementById('decay_input').value = trainer.l2_decay;
}
var toggle_pause = function() {
  pause = !pause;
  var btn = document.getElementById('buttontp');
  if(pause) { 
    btn.value = 'Resume';
  }
  else { 
    btn.value = 'Pause'; 
  }
}
var toggle_start = function () {
  for(var k=0;k<loaded.length;k++) { loaded[k] = false; }

  load_data_batch(0); // async load train set batch 0
  load_data_batch(test_batch); // async load test set
  var d_start = new Date();
  start_time = GetTimeString(d_start); //l???y th???i gian b???t ?????u hu???n luy???n
  start_fun();
}
var dump_json = function() {
  document.getElementById("dumpjson").value = JSON.stringify(this.net.toJSON());
}

var reset_all = function() {
  // reinit trainer
  trainer = new convnetjs.SGDTrainer(net, {learning_rate:trainer.learning_rate, momentum:trainer.momentum, batch_size:trainer.batch_size, l2_decay:trainer.l2_decay});
  update_net_param_display();

  // reinit windows that keep track of val/train accuracies
  xLossWindow.reset();
  wLossWindow.reset();
  trainAccWindow.reset();
  valAccWindow.reset();
  testAccWindow.reset();
  //lossGraph = new cnnvis.Graph(); // reinit graph too
  step_num = 0;
}
var load_from_json_snapshot = function() {
  var jsonString = document.getElementById("dumpjson").value;
  var json = JSON.parse(jsonString);
  net = new convnetjs.Net();
  net.fromJSON(json);
  reset_all();
}

//Load Json file
var loadJSON =  function(file, callback) {   
    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', file, true); // Replace 'my_data' with the path to your file
    xobj.onreadystatechange = function () {
          if (xobj.readyState == 4 && xobj.status == "200") {
            // Required use of an anonymous callback as .open will NOT return a value but simply returns undefined in asynchronous mode
            callback(xobj.responseText);
          }
    };
    xobj.send(null);  
}
//Load pretrained network and information
var load_network_pretrained = function() {
    var f = dataset_name+"_netowrk_snapshot.json"
    loadJSON(f, function(response) {
    var actual_JSON = JSON.parse(response);
    net = new convnetjs.Net();
    net.fromJSON(actual_JSON[0].network);
    trainer.learning_rate = actual_JSON[0].lr;
    trainer.momentum = actual_JSON[0].mmt;
    trainer.batch_size = actual_JSON[0].bsz;
    trainer.l2_decay = actual_JSON[0].wdc;
    reset_all();
    });
    console.log("finished load pretrained network");
}
//Save train informtaion to json file
var save_network_to_jsonfile = function(prefix){
  var _network = this.net.toJSON();
  var _learningRate = document.getElementById("lr_input").value;
  var _momentum = document.getElementById("momentum_input").value;
  var _batchSize = document.getElementById("batch_size_input").value;
  var _weightdecay = document.getElementById("decay_input").value;
  var data_network = [{network: _network, lr: _learningRate, mmt: _momentum, bsz: _batchSize, wdc: _weightdecay}];
  var contentfile = JSON.stringify(data_network);
  
  //save file
  var maxacc = parseFloat(testAccWindow.get_average()).toFixed(2)*100;
  var file_name = prefix + "_netowrk_snapshot_" + maxacc + ".json";
  SaveFiles(contentfile,file_name,"text/plain;charset=" + document.characterSet);
}
var manual_save_network_to_jsonfile = function(prefix){
  save_network_to_jsonfile(prefix);
  save_information_when_get_maxacc(start_time);
  save_infor_at_step(arrStepInfo);
}

//-------------------------------------X??? l?? l??u th??ng tin khi max acc-------------------------------------------
//Time max acc
//var time_train_finish;
var start_time;
var end_time;

//save info at step
var infor_at_step = function(stepnum, trainAccValue, validationAccValue, testAccvalue, Classification_loss_train)
{
  var objectInfoAtStep = new Object();
  objectInfoAtStep = [StepNumber = stepnum, Training_Accuracy = trainAccValue, Validation_Accuracy = validationAccValue, Testing_Accuracy = testAccvalue, Classification_loss = Classification_loss_train]
  return objectInfoAtStep;
}

var save_infor_at_step = function(data){
  var csvfile = 'Step Number, Training accuracy, Validation accuracy, Testing accuracy, Classification loss\n';
  for (var i = 0; i < data.length; i++) {
            //csvfile += data.join(',');
            csvfile += data[i];
            csvfile += "\n";
        }
  var maxacc = parseFloat(testAccWindow.get_average()*100).toFixed(4);
  var filename = dataset_name + "_Information_at_step_" + maxacc;
  var hiddenElement = document.createElement('a');
    hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csvfile);
    hiddenElement.target = '_blank';
    hiddenElement.download = filename+'.csv';
    hiddenElement.click();
}

//save infomation when get max acc
var save_information_when_get_maxacc = function(starttime){
  // Save information when get max acc
  var d_end = new Date();
  var maxacc = parseFloat(testAccWindow.get_average()*100).toFixed(4);
  end_time = GetTimeString(d_end);
  var f_name = dataset_name + "_Information_when_maxacc_" + maxacc + ".txt";

  var clasify_loss_train = parseFloat(xLossWindow.get_average()*100).toFixed(4);
  var weight_loss = parseFloat(wLossWindow.get_average()*100).toFixed(4);
  var validation_acc = parseFloat(valAccWindow.get_average()*100).toFixed(4);
  var train_accuracy = parseFloat(trainAccWindow.get_average()*100).toFixed(4);
  //json struct
 /* var datas_information = [{
                max_acc_train: maxacc,
                loss_max_acc_train: loss_max_acc,
                validation_acc_train: validation_acc,
                start_time_train: starttime,
                end_time_train: end_time,
               // interval_train: time_train_finish                  
              }];
    var contentSave = JSON.stringify(datas_information);*/
  var datas_information = '';
  datas_information += "Max_accuracy_Test_on_Testset: " + maxacc + " %"+'\n'+ "Training_accuracy: "+ train_accuracy +" %"+'\n'+ "Validation_acc_train:" + validation_acc + " %"+'\n'+ "Classification_loss_train: " + clasify_loss_train + " %" + '\n'+ "L2_Weight_decay_loss: "+ weight_loss + '\n' + "Examples_seen: " + step_num + '\n'+"Forward_time_per_example: "+ start_forward_time + '\n' + "Backprop_time_per_example: "+ backprop_time + '\n' + "Start_time_train: " + starttime  + '\n' + "End_time_train: " + end_time;
  SaveFiles(datas_information,f_name, "text/plain;charset=" + document.characterSet);
}
//save infomation when get time

// format time
var GetTimeString = function(dt){
  var month = dt.getMonth()+1;  
  var day = dt.getDate();  
  var year = dt.getFullYear();
  var hour = dt.getHours();
  var minute = dt.getMinutes();
  var second= dt.getSeconds();
  return day+"-"+ month + "-" + year + "---" + hour + "h " + minute + "m " + second + "s";
}
//save file
var SaveFiles = function(content, filename, type){
  var b = new Blob([content],{type:type});
  saveAs(b, filename);
}

//get  graph image
var Save_Image_From_Canvas = function(canvasID, imageName){
  var canvas = document.getElementById(canvasID);
  image = canvas.toDataURL('image/jpeg', 1.0);
  var maxacc = parseFloat(testAccWindow.get_average()).toFixed(2)*100;
  var event = new MouseEvent('click', {
    view: window,
    bubbles: true,
    cancelable: true
  });
  var aLink = document.createElement('a');
  aLink.download = dataset_name+ "_" + imageName + "_" + maxacc +'.png';
  aLink.href = image;
  console.log(aLink.href);
  aLink.dispatchEvent(event);
}

//Thay ?????i m???ng t??? textbox
var change_net = function() {
  eval($("#newnet").val());
  reset_all();
}
