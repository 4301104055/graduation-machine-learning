// int main
$(window).load(function() {
    load_network_pretrained_128(net_128_url);
    load_network_pretrained_information_128(net_128_info_url);
    load_network_pretrained_64L(net_64L_url);
    load_network_pretrained_information_64L(net_64L_info_url);
    load_network_pretrained_64R(net_64R_url);
    load_network_pretrained_information_64R(net_64R_info_url);
    //binding_image(image_list_url); 
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
    } 
    else {
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
var testImage = function(img) {
  var result_label_128;
  var result_value_128;
  var result_time_128;

  var obj128 = new Object();
  d = new Date();
  var start_time = d.getTime();
  console.log("Start Time: "+start_time);

  var x = convnetjs.img_to_vol(img);
  var out_p = net_128.forward(x);

  var vis_elt = document.getElementById("visnet");
  //visualize_activations(net_128, vis_elt);

  var preds =[]
  for(var k=0;k<out_p.w.length;k++) { preds.push({k:k,p:out_p.w[k]}); }
  preds.sort(function(a,b){return a.p<b.p ? 1:-1;});
  result_label_128 = classes_txt[preds[0].k];
  result_value_128 = preds[0].p;
  //Clear div before add result
  $("#testset_vis_128").empty();
  $("#testset_acc_128").empty();
  // add predictions
  var probsdiv = document.createElement('div');
  var t = '';
  var OneImgacc = '';
  for(var k=0;k<2;k++) {
    OneImgacc = parseFloat(preds[k].p/1*100).toFixed(5);
    //console.log (preds[k].k + "---test");
    var col = k===0 ? 'progress-bar bg-success':'progress-bar bg-danger';
    t +='<div style=\"float: left; margin-right: 15px; padding: 3px 3px 3px 3px; font-weight: bold;\">'+ classes_txt[preds[k].k]+'</div>'+'<div class=\"progress\"><div class=\"'+ col +'\" role=\"progressbar\" style=\"width:'+ OneImgacc + '%;\" aria-valuemin=\"0\" aria-valuemax=\"100\">'+ OneImgacc + ' %</div></div>'; 
  }
  d = new Date();
  var end_time = d.getTime();
  //console.log("End Time: "+end_time);
  var time = end_time - start_time;
  result_time_128 = time;
  probsdiv.innerHTML = t;
  probsdiv.className = 'probsdiv-test-one';

  $(probsdiv).prependTo($("#testset_vis_128")).hide().fadeIn('slow').slideDown('slow');
  $("#result-on-128image").text("CNN-128F Result");
  $("#testset_acc_128").text("----- Inference time: "+ time + " ms -----");

  obj128 = {
    l: result_label_128, 
    v: parseFloat(result_value_128).toFixed(5),
    t: result_time_128
  };
  return obj128;
}
var testImage_64L = function(img){
  var result_label_64L;
  var result_value_64L;
  var result_time_64L;
  var obj64L = new Object();
  d = new Date();
  var start_time = d.getTime();
  console.log("Start Time: "+start_time);

  var x = convnetjs.img_to_vol(img);
  var out_p = net_64L.forward(x);

  var vis_elt = document.getElementById("visnet");
  //visualize_activations(net_64L, vis_elt);

  var preds =[]
  for(var k=0;k<out_p.w.length;k++) { preds.push({k:k,p:out_p.w[k]}); }
  preds.sort(function(a,b){return a.p<b.p ? 1:-1;});
  result_label_64L = classes_txt[preds[0].k];
  result_value_64L = preds[0].p;
  //Clear div before add result
  $("#testset_vis_64L").empty();
  $("#testset_acc_64L").empty();
  // add predictions
  var probsdiv = document.createElement('div');
  var t = '';
  var OneImgacc = '';
  for(var k=0;k<2;k++) {
    OneImgacc = parseFloat(preds[k].p/1*100).toFixed(5);
    console.log (preds[k].k);
    var col = k===0 ? 'progress-bar bg-success':'progress-bar bg-danger';
    t +='<div style=\"float: left; margin-right: 15px; padding: 3px 3px 3px 3px; font-weight: bold;\">'+ classes_txt[preds[k].k]+'</div>'+'<div class=\"progress\"><div class=\"'+ col +'\" role=\"progressbar\" style=\"width:'+ OneImgacc + '%;\" aria-valuemin=\"0\" aria-valuemax=\"100\">'+ OneImgacc + ' %</div></div>'; 
  }
  d = new Date();
  var end_time = d.getTime();
  var time = end_time - start_time;
  result_time_64L = time;
  probsdiv.innerHTML = t;
  probsdiv.className = 'probsdiv-test-one';

  $(probsdiv).prependTo($("#testset_vis_64L")).hide().fadeIn('slow').slideDown('slow');
  $("#result-on-64L").text("CNN-64L Result:");
  $("#testset_acc_64L").text("----- Inference time: "+ time + " ms -----");
  obj64L = {
    l: result_label_64L, 
    v: parseFloat(result_value_64L).toFixed(5),
    t: result_time_64L
  };
  return obj64L;
}
var testImage_64R = function(img){
  var result_label_64R;
  var result_value_64R;
  var result_time_64R;
  d = new Date();
  var start_time = d.getTime();
  var obj64R = new Object();
  console.log("Start Time: "+start_time);

  var x = convnetjs.img_to_vol(img);
  var out_p = net_64R.forward(x);

  var vis_elt = document.getElementById("visnet");
  //visualize_activations(net_64L, vis_elt);

  var preds =[]
  for(var k=0;k<out_p.w.length;k++) { preds.push({k:k,p:out_p.w[k]}); }
  preds.sort(function(a,b){return a.p<b.p ? 1:-1;});
  result_label_64R = classes_txt[preds[0].k];
  result_value_64R = preds[0].p;
  //Clear div before add result
  $("#testset_vis_64R").empty();
  $("#testset_acc_64R").empty();
  // add predictions
  var probsdiv = document.createElement('div');
  var t = '';
  var OneImgacc = '';
  for(var k=0;k<2;k++) {
    OneImgacc = parseFloat(preds[k].p/1*100).toFixed(5);
    console.log (preds[k].k);
    var col = k===0 ? 'progress-bar bg-success':'progress-bar bg-danger';
    t +='<div style=\"float: left; margin-right: 15px; padding: 3px 3px 3px 3px; font-weight: bold;\">'+ classes_txt[preds[k].k]+'</div>'+'<div class=\"progress\"><div class=\"'+ col +'\" role=\"progressbar\" style=\"width:'+ OneImgacc + '%;\" aria-valuemin=\"0\" aria-valuemax=\"100\">'+ OneImgacc + ' %</div></div>'; 
  }
  d = new Date();
  var end_time = d.getTime();
  console.log("End Time: "+end_time);
  var time = end_time - start_time;
  result_time_64R = time;
  probsdiv.innerHTML = t;
  probsdiv.className = 'probsdiv-test-one';

  $(probsdiv).prependTo($("#testset_vis_64R")).hide().fadeIn('slow').slideDown('slow');
  $("#result-on-64R").text("CNN-64R Result:");
  $("#testset_acc_64R").text("----- Inference time: "+ time + " ms -----");
  obj64R = {
    l: result_label_64R, 
    v: parseFloat(result_value_64R).toFixed(5),
    t: result_time_64R
  };
  return obj64R;
}
var arrResultModelTest = [];
var test_cnn_model = function (img64L, img64R, img128){
  d = new Date();
  var start_time = d.getTime();
  var result_model_label;
  var final_conclusion_fr1 = "";
  var final_conclusion_fr2 = "";
  var final_conclusion_fr3 = "";
  var arrData;

  var obj = new Object();

  var result_64L =  testImage_64L(img64L);
  var result_64R =  testImage_64R(img64R);
  var result_128 =  testImage(img128);

  var type_tree = get_Type_tree(result_64L.l, result_64R.l, result_128.l);
  if (type_tree == 1)
    result_model_label = "Normal";
  if (type_tree == 2 || type_tree == 3 || type_tree == 4)
    result_model_label = "Consultation";
  if (type_tree == 5 || type_tree == 6 || type_tree == 7 )
    result_model_label = "Abnormal";
  if (type_tree == 8)
    result_model_label = "Consultation";

  final_conclusion_fr1 = Calculate_fr1(type_tree, result_64L.v, result_64R.v, result_128.v);
  final_conclusion_fr2 = Calculate_fr2(type_tree, result_64L.v, result_64R.v, result_128.v);
  final_conclusion_fr3 = Calculate_fr3(type_tree, result_64L.v, result_64R.v, result_128.v);

  d = new Date();
  var end_time = d.getTime();
  var time = end_time - start_time;
  
  $("#model-result-times").text("----- Inference time: "+ time + " ms -----");
  
  $("#model-result").text("");

  $("#model-result").text(result_model_label);

  if(type_tree == 2 || type_tree == 3 || type_tree == 4 || type_tree == 8){
    document.getElementById("title-result-fr").style.visibility = "visible";
    $("#model-result-value").text("- FR1 Conclusion: " + final_conclusion_fr1);
    $("#model-result-value").append('</br>');
    $("#model-result-value").append("- FR2 Conclusion: " +  final_conclusion_fr2);
    $("#model-result-value").append('</br>');
    $("#model-result-value").append("- FR3 Conclusion: " +  final_conclusion_fr3);
  }
  else{
    $("#model-result-value").text("");
    document.getElementById("title-result-fr").style.visibility = "hidden";
  }

  var image_file_name = $("#images_list option:selected").text();
  image_file_name = image_file_name.replace(/\n|\r/g, "");

  obj = [IMGL = image_file_name,
          L64 = result_64L.l, L64v = result_64L.v, Ltime = result_64L.t,
          R64 = result_64R.l, R64v = result_64R.v, Rtime = result_64R.t,
          F128 = result_128.l, F128v = result_128.v, Ftime = result_128.t,
          Mlabel= result_model_label, Mtype = type_tree, Mtime = time,
          fr1=final_conclusion_fr1, fr2 = final_conclusion_fr2, fr3=final_conclusion_fr3];
  arrResultModelTest.push(obj);
}

var Calculate_fr1 = function(type, value_64L, value_64R, value_128){
  var final_result = "";
  if(type == 2){
    var result_value = (value_128 + value_64R + (1-value_64L))/3;
    if(result_value > 0.5)
      final_result = "Normal";
    else
      final_result = "Abnormal";
  }
  if(type == 3){
    var result_value = (value_128 + value_64L + (1-value_64R))/3;
    if(result_value > 0.5)
      final_result = "Normal";
    else
      final_result = "Abnormal";
  }
  if(type == 4){
    var result_value = (value_128 + (1-value_64L) + (1-value_64R))/3;
    if(result_value > 0.5)
      final_result = "Normal";
    else
      final_result = "Abnormal";
  }
  if(type == 8){
    var result_value = (value_128 + (1-value_64L) + (1-value_64R))/3;
    if(result_value > 0.5)
      final_result = "Abnormal";
    else
      final_result = "Normal";
  }
  return final_result;
}

var Calculate_fr2 = function(type, value_64L, value_64R, value_128){
  var final_result = "";
  if(type == 2){
    var result_value = (value_128 + value_64R + (1-value_64L))/4;
    if(result_value > 0.5)
      final_result = "Normal";
    else
      final_result = "Abnormal";
  }
  if(type == 3){
    var result_value = (value_128 + value_64L + (1-value_64R))/4;
    if(result_value > 0.5)
      final_result = "Normal";
    else
      final_result = "Abnormal";
  }
  if(type == 4){
    var result_value = (value_128 + (1-value_64L) + (1-value_64R))/4;
    if(result_value > 0.5)
      final_result = "Normal";
    else
      final_result = "Abnormal";
  }
  if(type == 8){
    var result_value = (value_128 + (1-value_64L) + (1-value_64R))/4;
    if(result_value > 0.5)
      final_result = "Abnormal";
    else
      final_result = "Normal";
  }
  return final_result;
}

var Calculate_fr3 = function(type, value_64L, value_64R, value_128){
  var final_result = "";
  if(type == 2){
    var result_value = (value_128 + (1-value_64L))/2;
    if(result_value > 0.5)
      final_result = "Normal";
    else
      final_result = "Abnormal";
  }
  if(type == 3){
    var result_value = (value_128 + (1-value_64R))/2;
    if(result_value > 0.5)
      final_result = "Normal";
    else
      final_result = "Abnormal";
  }
  if(type == 4){
    var result_value = (2*value_128 + (1-value_64L) + (1-value_64R))/4;
    if(result_value > 0.5)
      final_result = "Normal";
    else
      final_result = "Abnormal";
  }
  if(type == 8){
    var result_value = (2*value_128 + (1-value_64L) + (1-value_64R))/4;
    if(result_value > 0.5)
      final_result = "Abnormal";
    else
      final_result = "Normal";
  }
  return final_result;
}

var get_Type_tree = function(label_64L, label_64R, label_128){
  var c0 = "Normal";
  var c1 = "Abnormal";
  var type_tree;
  if(label_128 == c0 && label_64L == c0 && label_64R == c0)
    return type_tree = 1;
  if(label_128 == c0 && label_64L == c0 && label_64R == c1)
    return type_tree = 2;
  if(label_128 == c0 && label_64L == c1 && label_64R == c0)
    return type_tree = 3;
  if(label_128 == c0 && label_64L == c1 && label_64R == c1)
    return type_tree = 4;
  if(label_128 == c1 && label_64L == c1 && label_64R == c1)
    return type_tree = 5;
  if(label_128 == c1 && label_64L == c0 && label_64R == c1)
    return type_tree = 6;
  if(label_128 == c1 && label_64L == c1 && label_64R == c0)
    return type_tree = 7;
  if(label_128 == c1 && label_64L == c0 && label_64R == c0)
    return type_tree = 8;
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
var net_128;
var load_network_pretrained_128 = function(file_ulr) {
    var f128 = file_ulr;
    loadJSON(f128, function(response) {
    var actual_JSON = JSON.parse(response);
    net_128 = new convnetjs.Net();
    net_128.fromJSON(actual_JSON[0].network);
    });
    console.log("Finished load pretrained network");
}
var net_64L;
var load_network_pretrained_64L = function(file_ulr) {
    var f64L = file_ulr;
    loadJSON(f64L, function(response) {
    var actual_JSON = JSON.parse(response);
    net_64L = new convnetjs.Net();
    net_64L.fromJSON(actual_JSON[0].network);
    });
    console.log("Finished load pretrained network");
}
var net_64R;
var load_network_pretrained_64R = function(file_ulr) {
    var f64R = file_ulr;
    loadJSON(f64R, function(response) {
    var actual_JSON = JSON.parse(response);
    net_64R = new convnetjs.Net();
    net_64R.fromJSON(actual_JSON[0].network);
    });
    console.log("Finished load pretrained network");
}
var load_network_pretrained_information_128 = function(file_url){
  var file_url = file_url;
  var xmlhttp = new XMLHttpRequest();
  var text_file_data;
  xmlhttp.onreadystatechange = function(){
    if(xmlhttp.status==200 && xmlhttp.readyState==4){    
        text_file_data = xmlhttp.responseText.split('\n');
          var max_acc_value = text_file_data[0].split(':');
          $('#max-acc-train-128').text("Max accuracy test on testset: "+ max_acc_value[1]);

          var training_accuracy = text_file_data[1].split(':');
          $('#training-acc-128').text("Training accuracy: "+ training_accuracy[1]);

          var validation_acc_value = text_file_data[2].split(':');
          $('#validation-acc-train-128').text("Validation accuracy in training: "+ validation_acc_value[1]);

          var classify_loss = text_file_data[3].split(':');
          $('#classify-loss-128').text("Classification loss: "+ classify_loss[1]);

          var weight_loss = text_file_data[4].split(':');
          $('#wloss-train-128').text("L2 Weight decay loss: "+ weight_loss[1]);

          var exam_seen = text_file_data[5].split(':');
          $('#exam-seen-128').text("Examples seen: "+ exam_seen[1]);

          var forward_time = text_file_data[6].split(':');
          $('#forward-time-128').text("Forward time per example: "+ forward_time[1] + " ms");

          var backprop_time = text_file_data[7].split(':');
          $('#backprop-time-128').text("Backprop time per example: "+ backprop_time[1] + " ms");

          var start_time_value = text_file_data[8].split(':');
          $('#start-training-time-128').text("Start training time: "+ start_time_value[1]);

          var end_time_value = text_file_data[9].split(':');
          $('#end-training-time-128').text("End training time: "+ end_time_value[1]);
    }
  }
  xmlhttp.open("GET",file_url,true);
  xmlhttp.send();
}
var load_network_pretrained_information_64L = function(file_url){
  var file_url = file_url;
  var xmlhttp = new XMLHttpRequest();
  var text_file_data;
  xmlhttp.onreadystatechange = function(){
    if(xmlhttp.status==200 && xmlhttp.readyState==4){    
        text_file_data = xmlhttp.responseText.split('\n');
          var max_acc_value = text_file_data[0].split(':');
          $('#max-acc-train-64L').text("Max accuracy test on testset: "+ max_acc_value[1]);

          var training_accuracy = text_file_data[1].split(':');
          $('#training-acc-64L').text("Training accuracy: "+ training_accuracy[1]);

          var validation_acc_value = text_file_data[2].split(':');
          $('#validation-acc-train-64L').text("Validation accuracy in training: "+ validation_acc_value[1]);

          var classify_loss = text_file_data[3].split(':');
          $('#classify-loss-64L').text("Classification loss: "+ classify_loss[1]);

          var weight_loss = text_file_data[4].split(':');
          $('#wloss-train-64L').text("L2 Weight decay loss: "+ weight_loss[1]);

          var exam_seen = text_file_data[5].split(':');
          $('#exam-seen-64L').text("Examples seen: "+ exam_seen[1]);

          var forward_time = text_file_data[6].split(':');
          $('#forward-time-64L').text("Forward time per example: "+ forward_time[1] + " ms");

          var backprop_time = text_file_data[7].split(':');
          $('#backprop-time-64L').text("Backprop time per example: "+ backprop_time[1] + " ms");

          var start_time_value = text_file_data[8].split(':');
          $('#start-training-time-64L').text("Start training time: "+ start_time_value[1]);

          var end_time_value = text_file_data[9].split(':');
          $('#end-training-time-64L').text("End training time: "+ end_time_value[1]);
    }
  }
  xmlhttp.open("GET",file_url,true);
  xmlhttp.send();
}
var load_network_pretrained_information_64R = function(file_url){
  var file_url = file_url;
  var xmlhttp = new XMLHttpRequest();
  var text_file_data;
  xmlhttp.onreadystatechange = function(){
    if(xmlhttp.status==200 && xmlhttp.readyState==4){    
        text_file_data = xmlhttp.responseText.split('\n');
          var max_acc_value = text_file_data[0].split(':');
          $('#max-acc-train-64R').text("Max accuracy test on testset: "+ max_acc_value[1]);

          var training_accuracy = text_file_data[1].split(':');
          $('#training-acc-64R').text("Training accuracy: "+ training_accuracy[1]);

          var validation_acc_value = text_file_data[2].split(':');
          $('#validation-acc-train-64R').text("Validation accuracy in training: "+ validation_acc_value[1]);

          var classify_loss = text_file_data[3].split(':');
          $('#classify-loss-64R').text("Classification loss: "+ classify_loss[1]);

          var weight_loss = text_file_data[4].split(':');
          $('#wloss-train-64R').text("L2 Weight decay loss: "+ weight_loss[1]);

          var exam_seen = text_file_data[5].split(':');
          $('#exam-seen-64R').text("Examples seen: "+ exam_seen[1]);

          var forward_time = text_file_data[6].split(':');
          $('#forward-time-64R').text("Forward time per example: "+ forward_time[1] + " ms");

          var backprop_time = text_file_data[7].split(':');
          $('#backprop-time-64R').text("Backprop time per example: "+ backprop_time[1] + " ms");

          var start_time_value = text_file_data[8].split(':');
          $('#start-training-time-64R').text("Start training time: "+ start_time_value[1]);

          var end_time_value = text_file_data[9].split(':');
          $('#end-training-time-64R').text("End training time: "+ end_time_value[1]);
    }
  }
  xmlhttp.open("GET",file_url,true);
  xmlhttp.send();
}
var save_model_result_to_csv = function(filename){
  save_test_result(arrResultModelTest, filename);
}
var binding_image = function (file_url){
  var xmlhttp = new XMLHttpRequest();
  var lines = [];
  var option='<option>' + "Select..." + '</option>';
  var length = 0;
  xmlhttp.onreadystatechange = function(){
  if(xmlhttp.status==200 && xmlhttp.readyState==4){    
      lines = xmlhttp.responseText.split('\n');
      length = lines.length;
      for(var i = 0; i<lines.length;i++){
          var echUrl = lines[i].split(':');
          var text = echUrl[2].split('/');
          option += '<option value="'+ lines[i] + '">' + text[3] + '</option>';
       }
       $('#images_list').find('option').remove();
       $('#images_list').append(option);
    };
  }
  xmlhttp.open("GET",file_url,true);
  xmlhttp.send(lines);
  return true;
}
var getimage = function(){
  var sp = $('#span_image');
  sp.empty();
  var img_preview_128_hide = document.createElement('img');
  img_preview_128_hide.className = "img-preview-full";
  img_preview_128_hide.id = "preview_img_128_hide";
  sp.prepend(img_preview_128_hide);

  var img_preview_64L_hide = document.createElement('img');
  img_preview_64L_hide.className = "img-preview-half";
  img_preview_64L_hide.id = "preview_img_64L_hide";
  sp.prepend(img_preview_64L_hide);

  var img_preview_64R_hide = document.createElement('img');
  img_preview_64R_hide.className = "img-preview-half";
  img_preview_64R_hide.id = "preview_img_64R_hide";
  sp.prepend(img_preview_64R_hide);

  var strUrl = $('#images_list').val();
  console.log(strUrl);

  var echUrl = strUrl.split(':');
  var storage_64L = echUrl[0];
  //console.log(storage_64L);
  var storage_64R = echUrl[1];
  //console.log(storage_64R);
  var storage_128 = echUrl[2];
 // console.log(storage_128);
  //var rhinoStorage = $('#images_list').val();
  var rhino_64L = $('#preview_img_64L_hide');
  var rhino_64R = $('#preview_img_64R_hide');
  var rhino_128 = $('#preview_img_128_hide');
  var image_prview_show_64L = $('#preview_img_64L_show');
  var image_prview_show_64R = $('#preview_img_64R_show');
  var image_prview_show_128 = $('#preview_img_128_show');

  if (strUrl) {

    image_prview_show_64L.attr("src", storage_64L);
    rhino_64L.attr("src", storage_64L);
    //rhino_64L.attr("src", resize_image(storage_64L,64,128));
    
    image_prview_show_64R.attr("src", storage_64R);
    rhino_64R.attr("src", storage_64R);
    //rhino_64R.attr("src", resize_image(storage_64R,64,128));

    image_prview_show_128.attr("src", storage_128);
    rhino_128.attr("src", storage_128);
    //rhino_128.attr("src", resize_image(storage_128,128,128));
  }
}
var save_test_result = function(data, filename){
  var csvfile = 'Image File Name, 64L-Label, 64L-value, 64L-Times (ms), 64R-Label, 64R-value, 64R-Times (ms), 128F-Label, 128F-value, 128F-Times (ms), Model Result, Model Type, Model Time (ms), FR1, FR2, FR3\n';
  for (var i = 0; i < data.length; i++) {
            csvfile += data[i];
            csvfile += "\n";
        }
  var hiddenElement = document.createElement('a');
    hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csvfile);
    hiddenElement.target = '_blank';
    hiddenElement.download = filename+'.csv';
    hiddenElement.click();
}
var processRow = function (row) {
            var finalVal = '';
            for (var j = 0; j < row.length; j++) {
                var innerValue = row[j] === null ? '' : row[j].toString();
                if (row[j] instanceof Date) {
                    innerValue = row[j].toLocaleString();
                };
                var result = innerValue.replace(/"/g, '""');
                if (result.search(/("|,|\n)/g) >= 0)
                    result = '"' + result + '"';
                if (j > 0)
                    finalVal += ',';
                finalVal += result;
            }
            return finalVal + '\n';
}
//save file
var SaveFiles = function(content, filename, type){
  var b = new Blob([content],{type:type});
  saveAs(b, filename);
}
//var data;
var resize_image = function (src, iw, ih) {

  var image_ori = new Image();
  image_ori.src = src;

  var canvas = document.createElement("canvas");
  var ctx = canvas.getContext("2d");
  ctx.clearRect(0,0, canvas.width, canvas.height);

  canvas.width = iw;
  canvas.height = ih;

  ctx.drawImage(image_ori, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg');
}
var getDatasetUrl = function(){
  var strDatasetUrl = $('#url_list').val();
  var datasetName = $('#url_list option:selected').text();
  image_list_url = strDatasetUrl;
  console.log(image_list_url);
  dataset_name = datasetName;
  console.log(dataset_name);
  //binding_image(image_list_url);
  var st = binding_image(image_list_url);
  if(st == true){
    alert("import successful!");
  }
  else
    alert("import not successful!");
}
/*------------xử lý auto test --->Chua xong---------
var autotestModel = function(){
  //var imgListLength = $('#images_list option').length;
  for(i=1; i<imgListLength;){
    $('#images_list').val($('#images_list option').eq(i).val());
    var selectedValue = $('#images_list').val();
    getimagefromUrl(selectedValue);
    var img64L = document.getElementById('preview_img_64L_hide');
    var img64R = document.getElementById('preview_img_64R_hide');
    var img128F = document.getElementById('preview_img_128_hide');
    //setTimeout(test_cnn_model(img64L,img64R,img128F),1000);
    test_cnn_model(img64L,img64R,img128F);
    setTimeout(i++, 10000);
    console.log(i);
  }
  binding_one_by_one_image();
  var img64L = document.getElementById('preview_img_64L_hide');
  var img64R = document.getElementById('preview_img_64R_hide');
  var img128F = document.getElementById('preview_img_128_hide');
  setTimeout(test_cnn_model(img64L,img64R,img128F),1000);
}
var binding_one_by_one_image = function(){
  var imgListLength = $('#images_list option').length;
  var i = 1
  while(i<imgListLength){
      $('#images_list').val($('#images_list option').eq(i).val());
      var selectedValue = $('#images_list').val();
      getimagefromUrl(selectedValue);
      i++;
    }
}
var getimagefromUrl = function(strUrl){
  var sp = $('#span_image');
  sp.empty();
  var img_preview_128_hide = document.createElement('img');
  img_preview_128_hide.className = "img-preview-full";
  img_preview_128_hide.id = "preview_img_128_hide";
  sp.prepend(img_preview_128_hide);

  var img_preview_64L_hide = document.createElement('img');
  img_preview_64L_hide.className = "img-preview-half";
  img_preview_64L_hide.id = "preview_img_64L_hide";
  sp.prepend(img_preview_64L_hide);

  var img_preview_64R_hide = document.createElement('img');
  img_preview_64R_hide.className = "img-preview-half";
  img_preview_64R_hide.id = "preview_img_64R_hide";
  sp.prepend(img_preview_64R_hide);

  //var strUrl = $('#images_list').val();
  //console.log(strUrl);

  var echUrl = strUrl.split(':');
  var storage_64L = echUrl[0];
  //console.log(storage_64L);
  var storage_64R = echUrl[1];
  //console.log(storage_64R);
  var storage_128 = echUrl[2];
 // console.log(storage_128);
  //var rhinoStorage = $('#images_list').val();
  var rhino_64L = $('#preview_img_64L_hide');
  var rhino_64R = $('#preview_img_64R_hide');
  var rhino_128 = $('#preview_img_128_hide');
  var image_prview_show_64L = $('#preview_img_64L_show');
  var image_prview_show_64R = $('#preview_img_64R_show');
  var image_prview_show_128 = $('#preview_img_128_show');

  if (strUrl) {

    image_prview_show_64L.attr("src", storage_64L);
    rhino_64L.attr("src", storage_64L);
    //rhino_64L.attr("src", resize_image(storage_64L,64,128));
    
    image_prview_show_64R.attr("src", storage_64R);
    rhino_64R.attr("src", storage_64R);
    //rhino_64R.attr("src", resize_image(storage_64R,64,128));

    image_prview_show_128.attr("src", storage_128);
    rhino_128.attr("src", storage_128);
    //rhino_128.attr("src", resize_image(storage_128,128,128));
  }
}
/*------------end xử lý auto test --->Chua xong---------*/

$(document).ready( function(){
  $('#images_list').change(function(){
      /*if($('#isAutoTest').prop('checked')){
        getimage();
        var img64L = document.getElementById('preview_img_64L_hide');
        var img64R = document.getElementById('preview_img_64R_hide');
        var img128F = document.getElementById('preview_img_128_hide');
        setTimeout(test_cnn_model(img64L,img64R,img128F),100000);
      }
      else{
        getimage();
      }*/
      getimage();
    });
});
