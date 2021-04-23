// int main
$(window).load(function() {
    $('#myModal').modal('show');
    var cbk128 = document.getElementById('128');
    var cbk64 = document.getElementById('64');
    var cbk32 = document.getElementById('32');
    $('#btnSave').click(function(){
        if(cbk128.checked == true){
          dataset_name = "x-rays_128";
          image_dimension = 128;
          file_url_net = dataset_name+"/"+ dataset_name +"_netowrk_snapshot_96.json";
          file_url_info = dataset_name+"/"+ dataset_name +"_Information_when_maxacc_96.txt";
          $('#myModal').modal('hide');
          waitingDialog.show("Loading pretrained network" + dataset_name);
          load_network_pretrained(file_url_net);
          load_network_pretrained_information(file_url_info),
          setTimeout(function () {waitingDialog.hide();}, 3000)
        }
        if(cbk64.checked == true){
          dataset_name = "x-rays_64";
          image_dimension = 64;
          file_url_net = dataset_name+"/"+ dataset_name +"_netowrk_snapshot_90.json";
          file_url_info = dataset_name+"/"+ dataset_name +"_Information_when_maxacc_90.txt";
          $('#myModal').modal('hide');
          waitingDialog.show("Loading pretrained network - " + dataset_name);
          load_network_pretrained(file_url_net);
          load_network_pretrained_information(file_url_info),
          setTimeout(function () {waitingDialog.hide();}, 3000)
        }
        if(cbk32.checked == true){
          //dataset_name = "x-rays_32";
          //image_dimension = 32;
          //file_url_net = dataset_name+"/"+ dataset_name +"_netowrk_snapshot_90.json";
          //file_url_info = dataset_name+"/"+ dataset_name +"_Information_when_maxacc_90.txt";
          //$('#myModal').modal('hide');
         // waitingDialog.show("Loading pretrained network - " + dataset_name);
          //load_network_pretrained(file_url_net);
          //load_network_pretrained_information(file_url_info),
          //setTimeout(function () {waitingDialog.hide();}, 3000)
          alert('No trained data, please choose another option!');
        }
    });
    $('#btnClose').click(function(){
      window.location="index.html";
    });
});

var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;
// elt is the element to add all the canvas activation drawings into
// A is the Vol() to use
// scale is a multiplier to make the visualizations larger. Make higher for larger pictures
// if grads is true then gradients are used instead
var draw_activations = function(elt, A, scale, grads) {

  var s = scale //|| 2; // scale
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
    var s = scale //|| 2; // scale
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
    var scale = 1; //chỉnh thông số hiển thị hình ảnh trong canvas
    if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax

    // HACK to draw in color in input layer
    if(i===0) {
      draw_activations_COLOR(activations_div, L.out_act, scale);
      draw_activations_COLOR(activations_div, L.out_act, scale, true);

      /*
      // visualize positive and negative components of the gradient separately
      var dd = L.out_act.clone();
      var ni = L.out_act.w.length;
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq > 0 ? dwq : 0.0; }
      draw_activations_COLOR(activations_div, dd, scale);
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq < 0 ? -dwq : 0.0; }
      draw_activations_COLOR(activations_div, dd, scale);
      */

      /*
      // visualize what the network would like the image to look like more
      var dd = L.out_act.clone();
      var ni = L.out_act.w.length;
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] -= 20*dwq; }
      draw_activations_COLOR(activations_div, dd, scale);
      */

      /*
      // visualize gradient magnitude
      var dd = L.out_act.clone();
      var ni = L.out_act.w.length;
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq*dwq; }
      draw_activations_COLOR(activations_div, dd, scale);
      */

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
      if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 1; // for softmax
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
var d; //Date object
var start_time;
var end_time;
var testImage = function(img) {
  d = new Date();
  start_time = d.getTime();
  console.log("Start Time: "+start_time);
  var x = convnetjs.img_to_vol(img);
  var out_p = net.forward(x);

  var vis_elt = document.getElementById("visnet");
  visualize_activations(net, vis_elt);

  var preds =[]
  for(var k=0;k<out_p.w.length;k++) { preds.push({k:k,p:out_p.w[k]}); }
  preds.sort(function(a,b){return a.p<b.p ? 1:-1;});

  //Clear div before add result
  $("#testset_vis").empty();
  $("#testset_acc").empty();
  // add predictions
  var probsdiv = document.createElement('div');
  var t = '';
  var OneImgacc = '';
  for(var k=0;k<2;k++) {
    OneImgacc = parseFloat(preds[k].p/1*100).toFixed(2);
    console.log (preds[k].k);
    var col = k===0 ? 'progress-bar bg-success':'progress-bar bg-danger';
    t +='<div style=\"float: left; margin-right: 15px; padding: 3px 3px 3px 3px; font-weight: bold;\">'+ classes_txt[preds[k].k]+'</div>'+'<div class=\"progress\"><div class=\"'+ col +'\" role=\"progressbar\" style=\"width:'+ OneImgacc + '%;\" aria-valuemin=\"0\" aria-valuemax=\"100\">'+ OneImgacc + ' %</div></div>'; 
  }
  d = new Date();
  end_time = d.getTime();
  console.log("End Time: "+end_time);
  var time = end_time - start_time;
  probsdiv.innerHTML = t;
  probsdiv.className = 'probsdiv-test-one';

  $(probsdiv).prependTo($("#testset_vis")).hide().fadeIn('slow').slideDown('slow');
  $("#testset_acc").text("----- Inference time: "+ time + " ms -----");
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
//Load pretrained
var load_network_pretrained = function(file_ulr) {
    var f = file_ulr;
    loadJSON(f, function(response) {
    var actual_JSON = JSON.parse(response);
    net = new convnetjs.Net();
    net.fromJSON(actual_JSON[0].network);
    });
    console.log("Finished load pretrained network");
}

var load_network_pretrained_information = function(file_url){
  var file_url = file_url;
  var xmlhttp = new XMLHttpRequest();
  var text_file_data;
  xmlhttp.onreadystatechange = function(){
    if(xmlhttp.status==200 && xmlhttp.readyState==4){    
        text_file_data = xmlhttp.responseText.split('\n');
          var max_acc_value = text_file_data[0].split(':');
          $('#max-acc-train').text("Max accuracy test on testset: "+ max_acc_value[1]);

          var training_accuracy = text_file_data[1].split(':');
          $('#training-acc').text("Training accuracy: "+ training_accuracy[1]);

          var validation_acc_value = text_file_data[2].split(':');
          $('#validation-acc-train').text("Validation accuracy in training: "+ validation_acc_value[1]);

          var classify_loss = text_file_data[3].split(':');
          $('#classify-loss').text("Classification loss: "+ classify_loss[1]);

          var weight_loss = text_file_data[4].split(':');
          $('#wloss-train').text("L2 Weight decay loss: "+ weight_loss[1]);

          var exam_seen = text_file_data[5].split(':');
          $('#exam-seen').text("Examples seen: "+ exam_seen[1]);

          var forward_time = text_file_data[6].split(':');
          $('#forward-time').text("Forward time per example: "+ forward_time[1] + " ms");

          var backprop_time = text_file_data[7].split(':');
          $('#backprop-time').text("Backprop time per example: "+ backprop_time[1] + " ms");

          var start_time_value = text_file_data[8].split(':');
          $('#start-training-time').text("Start training time: "+ start_time_value[1]);

          var end_time_value = text_file_data[9].split(':');
          $('#end-training-time').text("End training time: "+ end_time_value[1]);
    }
  }
  xmlhttp.open("GET",file_url,true);
  xmlhttp.send();
}
var read_text_file = function(file_url){
  var xmlhttp = new XMLHttpRequest();
  var lines;
  xmlhttp.onreadystatechange = function(){
    if(xmlhttp.status==200 && xmlhttp.readyState==4){    
        lines = xmlhttp.responseText.split('\n');
        for(i=0; i<lines.length; i++)
        {
          console.log(lines[i]);
        }
    }
  }
  xmlhttp.open("GET",file_url,true);
  xmlhttp.send(lines);
}
var getimage = function(){
  var sp = $('#span_image');
  sp.empty();
  var img_preview_hide = document.createElement('img');
  img_preview_hide.className = "img-preview";
  img_preview_hide.id = "preview_img";
  img_preview_hide.setAttribute("style","width:"+image_dimension+"px");
  img_preview_hide.setAttribute("style","height:"+image_dimension+"px");
  sp.prepend(img_preview_hide);
  var rhinoStorage = $('#images_list').val();
  var rhino = $('#preview_img');
  var image_prview_show = $('#preview_img_show')
  if (rhinoStorage) {
    // Reuse existing Data URL from localStorage
    rhino.attr("src", rhinoStorage);
    image_prview_show.attr("src", rhinoStorage)
  }
}

